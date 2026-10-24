"""Offline tests of active preparation, inference helpers and summaries."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_kie", "src/step_01_prepare/prepare_kie.py")
runner = load("qwen35_runner", "src/step_02_vlm/qwen35/run.py")
summarize = load("summarize", "src/step_04_evaluation/summarize_all.py")


class Qwen35OfflineTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"dataset_name": "POIE", "type": "key information extraction en", "id": i,
             "image_path": f"EN_part/a{i}.jpg", "question": "Extract the key",
             "answers": ["{'A': ['x']}"]}
            for i in range(12)
        ] + [
            {"dataset_name": "POIE", "type": "key information extraction cn", "id": 80,
             "image_path": "CN_part/a.jpg", "question": "question"}
        ]

    def test_prepare_selects_only_english_kie(self):
        selected = prepare.select_kie_examples(self.rows, dataset_name="poie", limit=4, seed=42)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(r["type"] == "key information extraction en" for r in selected))
        self.assertEqual(selected, prepare.select_kie_examples(self.rows, "POIE", 4, 42))

    def test_image_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "OCRBench_v2"
            image = root / "EN_part" / "a.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"test")
            self.assertEqual(runner.resolve_image(root, "EN_part/a.jpg"), image.resolve())
            self.assertEqual(runner.resolve_image(root, "OCRBench_v2/EN_part/a.jpg"), image.resolve())
            with self.assertRaises(FileNotFoundError):
                runner.resolve_image(root, "EN_part/missing.jpg")

    def test_prediction_identity_and_thinking_removal(self):
        self.assertNotEqual(runner.record_id(self.rows[0]), runner.record_id(self.rows[1]))
        self.assertEqual(runner.strip_reasoning("<think>working</think>\n{'SS':'122g'}"), "{'SS':'122g'}")
        self.assertEqual(runner.strip_reasoning("<think>unfinished</think>\n{}"), "{}")

    def test_atomic_benchmark_list_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prediction.json"
            rows = [{**self.rows[0], "predict": "{'A':'x'}", "inference_seconds": 1.5, "score": 1}]
            runner.write_json_atomic(path, rows)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), rows)
            metrics = summarize.summarize(rows)
            self.assertEqual(metrics["mean_official_kie_f1"], 1)
            self.assertEqual(metrics["median_inference_seconds"], 1.5)


if __name__ == "__main__":
    unittest.main()
