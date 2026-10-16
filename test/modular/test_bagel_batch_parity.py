"""CPU tests for the paired BAGEL serial/batched result comparator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "experiments/bagel/scripts"
sys.path.insert(0, str(SCRIPTS))

from compare_batch_parity import compare_images, trace_evidence  # noqa: E402


class BatchParityComparatorTest(unittest.TestCase):
    def _row(self, path: Path, sha256: str) -> dict:
        return {
            "request_index": 0,
            "prompt": "test prompt",
            "seed": 17,
            "width": 2,
            "height": 2,
            "image": {"path": str(path), "sha256": sha256},
        }

    def test_identical_images_report_exact_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.fromarray(np.full((2, 2, 3), 42, dtype=np.uint8)).save(path)

            result = compare_images(
                self._row(path, "same"), self._row(path, "same")
            )

        self.assertTrue(result["sha256_equal"])
        self.assertEqual(result["max_abs_pixel_error"], 0)
        self.assertEqual(result["equal_pixel_fraction"], 1)
        self.assertEqual(result["psnr_db"], float("inf"))

    def test_trace_requires_different_per_request_loop_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_dir = root / "execution_trace"
            trace_dir.mkdir()
            rows = [
                {
                    "event_type": "component_complete",
                    "node_name": "LLM",
                    "graph_walk": "image_gen",
                    "batch_id": 1,
                    "batch_size": 2,
                    "request_ids": ["a", "b"],
                    "dynamic_loop_iter_counts": {
                        "1": {"image_gen_loop": 3},
                        "2": {"image_gen_loop": 20},
                    },
                },
                {
                    "event_type": "component_complete",
                    "node_name": "LLM",
                    "graph_walk": "image_gen",
                    "batch_id": 2,
                    "batch_size": 2,
                    "request_ids": ["c", "d"],
                    "dynamic_loop_iter_counts": {
                        "3": {"image_gen_loop": 4},
                        "4": {"image_gen_loop": 4},
                    },
                },
            ]
            (trace_dir / "worker.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

            evidence = trace_evidence(root)

        self.assertEqual(evidence["image_component_completions"], 2)
        self.assertEqual(evidence["multi_request_image_completions"], 2)
        self.assertEqual(evidence["heterogeneous_step_image_completions"], 1)
        self.assertEqual(evidence["maximum_image_batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
