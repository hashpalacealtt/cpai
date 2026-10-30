import unittest
from unittest.mock import Mock, patch

from cpai.benchmark import parse_size_candidates, parse_thread_candidates
from cpai.engine import EngineError, LlamaCppEngine, extract_metrics


class MetricsTests(unittest.TestCase):
    def test_extracts_backend_timings(self):
        result = extract_metrics({
            "timings": {
                "predicted_n": 100,
                "predicted_ms": 5000,
                "predicted_per_second": 20.0,
                "prompt_n": 50,
                "prompt_ms": 1000,
                "prompt_per_second": 50.0,
            }
        })
        self.assertEqual(result["generated_tokens"], 100)
        self.assertEqual(result["generation_tokens_per_second"], 20.0)
        self.assertEqual(result["prompt_tokens_per_second"], 50.0)

    def test_computes_rates_when_backend_omits_them(self):
        result = extract_metrics({
            "timings": {
                "predicted_n": 40,
                "predicted_ms": 2000,
                "prompt_n": 20,
                "prompt_ms": 500,
            }
        })
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


class EngineRequestTests(unittest.TestCase):
    def test_default_command_does_not_enable_speculative_decoding(self):
        engine = LlamaCppEngine(port=0)
        command = engine._build_command("llama-server")
        self.assertNotIn("--spec-type", command)
        self.assertNotIn("--spec-draft-hf", command)

    def test_speculative_draft_command_uses_draft_simple_and_thread_budget(self):
        engine = LlamaCppEngine(
            port=0,
            threads=2,
            draft_model="Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M",
            draft_tokens=8,
            draft_threads=1,
        )
        command = engine._build_command("llama-server")
        self.assertEqual(command[command.index("--spec-type") + 1], "draft-simple")
        self.assertEqual(command[command.index("--spec-draft-hf") + 1],
                         "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "8")
        self.assertEqual(command[command.index("--spec-draft-threads") + 1], "1")

    def test_rejects_zero_draft_tokens(self):
        with self.assertRaises(ValueError):
            LlamaCppEngine(port=0, draft_tokens=0)

    def test_completion_can_ignore_eos_for_fixed_length_benchmarks(self):
        engine = LlamaCppEngine(port=0)
        process = Mock()
        process.poll.return_value = None
        engine.process = process
        try:
            with patch("cpai.engine._request_json", return_value={"content": "ok"}) as request:
                result = engine.complete("benchmark prompt", max_tokens=256, ignore_eos=True)
            self.assertEqual(result["content"], "ok")
            args, kwargs = request.call_args
            self.assertEqual(args[0], f"{engine.base_url}/completion")
            self.assertEqual(args[1]["n_predict"], 256)
            self.assertTrue(args[1]["ignore_eos"])
            self.assertEqual(args[1]["temperature"], 0.0)
            self.assertFalse(args[1]["cache_prompt"])
            self.assertEqual(kwargs["timeout"], 600)
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
