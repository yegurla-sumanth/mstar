"""CPU contracts for the pruned batching gate; no GPU performance claims."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/bagel/scripts"))
from analyze_pruned_batch_gate import trace_contract, compare_latents, CAPTURE_STEPS
import profile_staggered


def event(ids, ages):
    return {
        "event_type": "component_complete", "node_name": "LLM", "graph_walk": "image_gen",
        "batch_size": len(ids), "request_ids": ids, "request_handles": list(range(len(ids))),
        "dynamic_loop_iter_counts": {str(i): {"image_gen_loop": age} for i, age in enumerate(ages)},
        "model_metadata": {
            "cfg_active_labels_by_request_handle": {str(i): ["main", "cfg_text"] for i in range(len(ids))},
            "cfg_virtual_rows_by_request_handle": {str(i): 2 for i in range(len(ids))},
        }, "cuda_duration_ms": 10,
    }


class GateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "execution_trace").mkdir()
        self.rows = [event(["a"], [0])]
        self.rows += [event(["a", "b"], [step, step - 1]) for step in range(1, 49)]
        self.rows += [event(["b"], [48])]

    def check(self, rows=None, cap=2):
        (self.root / "execution_trace/worker.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in (rows if rows is not None else self.rows)))
        return trace_contract(self.root, ["a", "b"], cap)

    def test_full_measured_heterogeneous_contract(self):
        report = self.check()
        self.assertEqual(report["heterogeneous_batch_count"], 48)
        self.assertEqual(report["virtual_rows_per_request"], {"a": 98, "b": 98})

    def test_missing_step_rejected(self):
        with self.assertRaisesRegex(ValueError, "0..48"):
            self.check(self.rows[:-1])

    def test_duplicate_step_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.check(self.rows + self.rows[:1])

    def test_wrong_branch_work_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["model_metadata"]["cfg_virtual_rows_by_request_handle"]["0"] = 3
        with self.assertRaisesRegex(ValueError, "two drop_image"):
            self.check(rows)

    def test_b1_cannot_contain_b2_batch(self):
        with self.assertRaisesRegex(ValueError, "membership/size"):
            self.check(cap=1)

    def test_warmup_only_mixed_batch_not_accepted(self):
        rows = [event([key], [step]) for key in ("a", "b") for step in range(49)]
        rows.append(event(["warmup-a", "warmup-b"], [10, 0]))
        with self.assertRaisesRegex(ValueError, "No measured heterogeneous"):
            self.check(rows)

    def test_same_step_batch_not_accepted_as_heterogeneous(self):
        with self.assertRaisesRegex(ValueError, "No measured heterogeneous"):
            self.check([event(["a", "b"], [step, step]) for step in range(49)])

    def test_cli_matches_orchestrator(self):
        with patch.object(sys, "argv", ["profile_staggered.py", "--output-dir", str(self.root),
                                        "--cfg-branch-policy", "drop_image", "--warmup-requests", "2",
                                        "--base-seed", "20261016"]):
            args = profile_staggered.parse_args()
        self.assertEqual(args.cfg_branch_policy, "drop_image")
        self.assertEqual(args.warmup_requests, 2)

    def test_profile_request_sends_frozen_cfg_workload(self):
        captured = {}
        request_dir = self.root / "requests" / "request-a"

        class Client:
            def __init__(self, *_args, **_kwargs):
                self.assertion = request_dir.exists()

            def generate(self, **kwargs):
                captured.update(kwargs)
                captured["directory_existed_before_generate"] = self.assertion
                raise RuntimeError("expected test stop")

        class Events:
            def write(self, *_args, **_kwargs):
                pass

        fake_mstar = types.SimpleNamespace(MStarClient=Client)
        with patch.dict(sys.modules, {"mstar": fake_mstar}):
            row = profile_staggered.run_request(
                base_url="http://unused", output_dir=self.root,
                request_id="request-a", request_index=0, phase="measured",
                prompt="fixed prompt", seed=17, width=1024, height=1024,
                timeout_seconds=1, target_perf_ns=time.perf_counter_ns(),
                events=Events(), cfg_branch_policy="drop_image",
            )
        self.assertEqual(row["status"], "failed")
        self.assertTrue(captured["directory_existed_before_generate"])
        self.assertEqual(captured["cfg_branch_policy"], "drop_image")
        self.assertEqual(captured["cfg_text_scale"], 4.0)
        self.assertEqual(captured["cfg_img_scale"], 1.0)
        self.assertEqual(captured["cfg_interval"], [0.4, 1.0])

    def test_real_torch_latents_exact_drift_missing_and_nonfinite(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable locally; this test runs in the pinned Colab venv")
        for name in ("serial", "batch"):
            directory = self.root / name / "debug_latents" / "a"
            directory.mkdir(parents=True)
            for step in CAPTURE_STEPS:
                torch.save({"request_id": "a", "step_index": step, "cfg_branch_policy": "drop_image",
                            "latents": torch.ones(2, 3)}, directory / f"after_step_{step:02d}.pt")
        args = (self.root / "serial", self.root / "batch", {"request_id": "a"}, {"request_id": "a"})
        self.assertTrue(all(row["exact"] for row in compare_latents(*args)))
        path = self.root / "batch/debug_latents/a/after_step_48.pt"
        payload = torch.load(path, weights_only=True)
        payload["latents"][0, 0] = 2
        torch.save(payload, path)
        self.assertEqual(compare_latents(*args)[-1]["max_abs_error"], 1)
        payload["latents"][0, 0] = float("nan")
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            compare_latents(*args)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            compare_latents(*args)


if __name__ == "__main__":
    unittest.main()
