"""CPU inference engine backed by a locally built llama.cpp server."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"
_CACHE_TYPES = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}


class EngineError(RuntimeError):
    """Raised when the inference backend cannot be started or queried."""


def find_server() -> str:
    candidate = os.environ.get("CPAI_LLAMA_SERVER", "llama-server")
    resolved = shutil.which(candidate)
    if not resolved and Path(candidate).is_file():
        resolved = str(Path(candidate).resolve())
    if not resolved:
        raise EngineError(
            "llama-server was not found. Install llama.cpp and put llama-server "
            "on PATH, or set CPAI_LLAMA_SERVER to its executable path."
        )
    return resolved


def _request_json(
    url: str, payload: dict[str, Any] | None = None, timeout: float = 30
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise EngineError(f"llama-server request failed: {exc}") from exc


def extract_metrics(response: dict[str, Any]) -> dict[str, float | int]:
    """Normalize llama.cpp completion timing fields."""
    timings = response.get("timings") or {}
    generated = int(timings.get("predicted_n", 0))
    generation_ms = float(timings.get("predicted_ms", 0.0))
    prompt_tokens = int(timings.get("prompt_n", 0))
    prompt_ms = float(timings.get("prompt_ms", 0.0))
    generation_tps = (
        float(timings["predicted_per_second"])
        if timings.get("predicted_per_second") is not None
        else (generated * 1000.0 / generation_ms if generation_ms > 0 else 0.0)
    )
    prompt_tps = (
        float(timings["prompt_per_second"])
        if timings.get("prompt_per_second") is not None
        else (prompt_tokens * 1000.0 / prompt_ms if prompt_ms > 0 else 0.0)
    )
    if generated <= 0 or generation_tps <= 0:
        raise EngineError(f"Backend returned no usable generation timings: {timings!r}")
    return {
        "generated_tokens": generated,
        "generation_ms": round(generation_ms, 3),
        "generation_tokens_per_second": round(generation_tps, 3),
        "prompt_tokens": prompt_tokens,
        "prompt_ms": round(prompt_ms, 3),
        "prompt_tokens_per_second": round(prompt_tps, 3),
    }


class LlamaCppEngine:
    """Manage llama-server with reusable prompt state and configurable CPU memory paths."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        threads: int = 4,
        threads_batch: int | None = None,
        batch_size: int = 512,
        ubatch_size: int = 128,
        context_size: int = 2048,
        port: int = 18080,
        startup_timeout: int = 900,
        draft_model: str | None = None,
        draft_tokens: int = 8,
        draft_threads: int | None = None,
        cache_prompt: bool = True,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        flash_attention: str = "auto",
        numa: str | None = None,
        mlock: bool = False,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be at least 1")
        if threads_batch is not None and threads_batch < 1:
            raise ValueError("threads_batch must be at least 1")
        if batch_size < 1 or ubatch_size < 1 or ubatch_size > batch_size:
            raise ValueError("require 1 <= ubatch_size <= batch_size")
        if draft_tokens < 1:
            raise ValueError("draft_tokens must be at least 1")
        if draft_threads is not None and draft_threads < 1:
            raise ValueError("draft_threads must be at least 1")
        if cache_type_k not in _CACHE_TYPES or cache_type_v not in _CACHE_TYPES:
            raise ValueError(f"KV cache types must be one of: {', '.join(sorted(_CACHE_TYPES))}")
        if flash_attention not in {"auto", "on", "off"}:
            raise ValueError("flash_attention must be auto, on, or off")
        if numa not in {None, "distribute", "isolate", "numactl"}:
            raise ValueError("numa must be distribute, isolate, numactl, or None")

        self.model = model
        self.threads = threads
        self.threads_batch = threads_batch or threads
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.context_size = context_size
        self.draft_model = draft_model
        self.draft_tokens = draft_tokens
        self.draft_threads = draft_threads or max(1, threads // 2)
        self.cache_prompt = cache_prompt
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.flash_attention = flash_attention
        self.numa = numa
        self.mlock = mlock
        self.port = self._free_port(port)
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[str] | None = None
        self._log_file: tempfile._TemporaryFileWrapper | None = None
        self.base_url = f"http://127.0.0.1:{self.port}"

    @staticmethod
    def _free_port(preferred: int) -> int:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                sock.bind(("127.0.0.1", 0))
                return int(sock.getsockname()[1])

    def _build_command(self, executable: str) -> list[str]:
        command = [
            executable,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--threads", str(self.threads),
            "--threads-batch", str(self.threads_batch),
            "--batch-size", str(self.batch_size),
            "--ubatch-size", str(self.ubatch_size),
            "--n-gpu-layers", "0",
            "--ctx-size", str(self.context_size),
            "--flash-attn", self.flash_attention,
            "--cache-type-k", self.cache_type_k,
            "--cache-type-v", self.cache_type_v,
            "--no-webui",
            "-hf", self.model,
        ]
        if self.numa:
            command.extend(["--numa", self.numa])
        if self.mlock:
            command.append("--mlock")
        if self.draft_model:
            command.extend([
                "--spec-type", "draft-simple",
                "--spec-draft-hf", self.draft_model,
                "--spec-draft-n-max", str(self.draft_tokens),
                "--spec-draft-threads", str(self.draft_threads),
                "--spec-draft-threads-batch", str(self.draft_threads),
            ])
        return command

    def start(self) -> None:
        if self.process is not None:
            return
        executable = find_server()
        self._log_file = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8")
        command = self._build_command(executable)
        server_env = os.environ.copy()
        server_env.setdefault("OMP_NUM_THREADS", str(max(self.threads, self.threads_batch)))
        server_env.setdefault("OMP_DYNAMIC", "FALSE")
        server_env.setdefault("OMP_PROC_BIND", "close")
        server_env.setdefault("OMP_PLACES", "cores")
        self.process = subprocess.Popen(
            command,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=server_env,
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise EngineError(f"llama-server exited during startup. Logs:\n{self.logs()}")
            try:
                _request_json(f"{self.base_url}/health", timeout=3)
                return
            except EngineError:
                time.sleep(1)
        raise EngineError(f"Timed out waiting for llama-server. Logs:\n{self.logs()}")

    def complete(
        self, prompt: str, max_tokens: int = 128, *, ignore_eos: bool = False,
        cache_prompt: bool | None = None,
    ) -> dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            raise EngineError("Engine is not running; call start() first.")
        return _request_json(
            f"{self.base_url}/completion",
            {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": 0.0,
                "cache_prompt": self.cache_prompt if cache_prompt is None else cache_prompt,
                "ignore_eos": ignore_eos,
                "stop": [],
            },
            timeout=600,
        )

    def logs(self) -> str:
        if self._log_file is None:
            return ""
        self._log_file.flush()
        self._log_file.seek(0)
        return self._log_file.read()[-12000:]

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "LlamaCppEngine":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
