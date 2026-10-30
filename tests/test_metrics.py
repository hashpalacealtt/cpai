import unittest
from unittest.mock import Mock, patch

from cpai.benchmark import (
    parse_choice_candidates, parse_size_candidates, parse_thread_candidates,
)
from cpai.engine import EngineError, LlamaCppEngine, extract_metrics


class MetricsTests(unittest.TestCase):
    def test_extracts_backend_timings(self):
        result = extract_metrics({"timings": {
            "predicted_n": 100, "predicted_ms": 5000,
            "predicted_per_second": 20.0, "prompt_n": 50,
            "prompt_ms": 1000, "prompt_per_second": 50.0,
        }})
        self.assertEqual(result["generated_tokens"], 100)
        self.assertEqual(result["generation_tokens_per_second"], 20.0)
        self.assertEqual(result["prompt_tokens_per_second"], 50.0)

    def test_computes_rates_when_backend_omits_them(self):
        result = extract_metrics({"timings": {
            "predicted_n": 40, "predicted_ms": 2000,
            "prompt_n": 20, "prompt_ms": 500,
        }})
        self.assertEqual(result["generation_tokens_per_second"], 20.0)
        self.assertEqual(result["prompt_tokens_per_second"], 40.0)

    def test_rejects_missing_generation_timing(self):
        with self.assertRaises(EngineError):
            extract_metrics({"timings": {"predicted_n": 0, "predicted_ms": 0}})

    def test_thread_candidates_are_sorted_and_deduplicated(self):
        self.assertEqual(parse_thread_candidates("4,2,4,1", 4), [1, 2, 4])

    def test_thread_candidates_reject_out_of_range_values(self):
        with self.assertRaises(Exception):
            parse_thread_candidates("1,5", 4)

    def test_size_candidates_are_sorted_and_deduplicated(self):
        self.assertEqual(parse_size_candidates("512,256,512", "batch"), [256, 512])

    def test_size_candidates_reject_non_positive_values(self):
        with self.assertRaises(Exception):
            parse_size_candidates("0,128", "ubatch")

    def test_choice_candidates_validate_and_deduplicate(self):
        self.assertEqual(
            parse_choice_candidates("ngram-mod,none,none", {"none", "ngram-mod"}, "spec"),
            ["ngram-mod", "none"],
        )
        with self.assertRaises(Exception):
            parse_choice_candidates("turboquant", {"none", "ngram-mod"}, "spec")


class EngineCommandTests(unittest.TestCase):
    def test_default_command_has_runtime_cache_and_cpu_scheduling(self):
        engine = LlamaCppEngine(port=0)
        command = engine._build_command("llama-server")
        self.assertIn("--flash-attn", command)
        self.assertEqual(command[command.index("--flash-attn") + 1], "auto")
        self.assertEqual(command[command.index("--cache-type-k") + 1], "f16")
        self.assertEqual(command[command.index("--cpu-strict") + 1], "1")
        self.assertEqual(command[command.index("--poll") + 1], "100")
        self.assertEqual(command[command.index("--spec-type") + 1], "ngram-mod")
        self.assertNotIn("--mlock", command)

    def test_cpu_affinity_and_polling_are_configurable(self):
        engine = LlamaCppEngine(port=0, cpu_strict=True, poll=0)
        command = engine._build_command("llama-server")
        self.assertEqual(command[command.index("--cpu-strict") + 1], "1")
        self.assertEqual(command[command.index("--poll") + 1], "0")

    def test_optional_memory_flags(self):
        engine = LlamaCppEngine(
            port=0, numa="distribute", mlock=True,
            cache_type_k="q8_0", cache_type_v="q8_0",
        )
        command = engine._build_command("llama-server")
        self.assertIn("--numa", command)
        self.assertIn("--mlock", command)
        self.assertEqual(command[command.index("--cache-type-v") + 1], "q8_0")

    def test_rejects_invalid_cache_types(self):
        with self.assertRaises(ValueError):
            LlamaCppEngine(port=0, cache_type_k="turboquant")

    def test_draft_model_command_uses_separate_thread_budget(self):
        engine = LlamaCppEngine(
            port=0, threads=2,
            draft_model="Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M",
            draft_tokens=8, draft_threads=1, speculative_type="none",
        )
        command = engine._build_command("llama-server")
        self.assertEqual(command[command.index("--spec-type") + 1], "draft-simple")
        self.assertEqual(command[command.index("--spec-draft-threads") + 1], "1")

    def test_ngram_speculative_command_is_native_and_model_free(self):
        engine = LlamaCppEngine(port=0, speculative_type="ngram-mod")
        command = engine._build_command("llama-server")
        self.assertEqual(command[command.index("--spec-type") + 1], "ngram-mod")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "64")
        self.assertEqual(command[command.index("--spec-ngram-mod-n-match") + 1], "24")

    def test_rejects_invalid_poll_and_speculative_modes(self):
        with self.assertRaises(ValueError):
            LlamaCppEngine(port=0, poll=101)
        with self.assertRaises(ValueError):
            LlamaCppEngine(port=0, speculative_type="turboquant")

    def test_completion_uses_prompt_cache_by_default_and_can_disable_it(self):
        engine = LlamaCppEngine(port=0)
        process = Mock()
        process.poll.return_value = None
        engine.process = process
        try:
            with patch("cpai.engine._request_json", return_value={"content": "ok"}) as request:
                engine.complete("prompt", max_tokens=256, ignore_eos=True)
                self.assertTrue(request.call_args.args[1]["cache_prompt"])
                engine.complete("prompt", cache_prompt=False)
                self.assertFalse(request.call_args.args[1]["cache_prompt"])
        finally:
            engine.process = None

    def test_completion_keeps_eos_enabled_by_default(self):
        engine = LlamaCppEngine(port=0)
        process = Mock()
        process.poll.return_value = None
        engine.process = process
        try:
            with patch("cpai.engine._request_json", return_value={"content": "ok"}) as request:
                engine.complete("normal prompt")
                self.assertFalse(request.call_args.args[1]["ignore_eos"])
        finally:
            engine.process = None


if __name__ == "__main__":
    unittest.main()
