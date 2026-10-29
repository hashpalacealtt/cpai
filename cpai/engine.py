"""Small stdlib-only client for a local llama.cpp server."""

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


def _request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30) -> dict[str, Any]:
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
    """Manage a local llama-server process and expose text completion."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        threads: int = 4,
        context_size: int = 2048,
        port: int = 18080,
        startup_timeout: int = 900,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be at least 1")
        self.model = model
        self.threads = threads
        self.context_size = context_size
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

    def start(self) -> None:
        if self.process is not None:
            return
        executable = find_server()
        self._log_file = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8")
        command = [
            executable,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--threads", str(self.threads),
            "--n-gpu-layers", "0",
            "--ctx-size", str(self.context_size),
            "--no-webui",
            "-hf", self.model,
        ]
        self.process = subprocess.Popen(
            command,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise EngineError(f"llama-server exited during startup. Logs:\n{self.logs()}")
            try:
                _request_json(f"{self.base_url}/health", timeout=3)
                return
            except EngineError:
                time.sleep(2)
        raise EngineError(f"Timed out waiting for llama-server. Logs:\n{self.logs()}")

    def complete(self, prompt: str, max_tokens: int = 128) -> dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            raise EngineError("Engine is not running; call start() first.")
        return _request_json(
            f"{self.base_url}/completion",
            {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": 0.0,
                "cache_prompt": False,
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
