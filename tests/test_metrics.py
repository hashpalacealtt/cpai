import unittest

from cpai.benchmark import parse_thread_candidates
from cpai.engine import EngineError, extract_metrics


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


if __name__ == "__main__":
    unittest.main()
