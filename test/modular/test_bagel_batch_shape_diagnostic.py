"""CPU contracts for the BAGEL B=1/homogeneous-B=2/heterogeneous-B=2 diagnostic."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/bagel/scripts"))

from analyze_batch_shape_diagnostic import (
    CAPTURE_STEPS,
    MIN_HOMOGENEOUS_BATCHES,
    command_stdout,
    compare_trajectory,
    trace_contract,
)
import run_profile


def event(ids: list[str], ages: list[int], batch_id: int = 1) -> dict:
    handles = list(range(len(ids)))
    return {
        "event_type": "component_complete",
        "node_name": "LLM",
        "graph_walk": "image_gen",
        "batch_id": batch_id,
        "batch_size": len(ids),
        "request_ids": ids,
        "request_handles": handles,
        "dynamic_loop_iter_counts": {
            str(handle): {"image_gen_loop": age}
            for handle, age in zip(handles, ages, strict=True)
        },
        "model_metadata": {
            "cfg_active_labels_by_request_handle": {
                str(handle): ["main", "cfg_text"] for handle in handles
            },
            "cfg_virtual_rows_by_request_handle": {
                str(handle): 2 for handle in handles
            },
        },
        "cuda_duration_ms": 10.0,
    }


class BatchShapeDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "execution_trace").mkdir()

    def write(self, rows: list[dict]) -> None:
        (self.root / "execution_trace/worker.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    def test_recorded_git_head_is_extracted_from_command_result(self):
        payload = {
            "command": ["git", "rev-parse", "HEAD"],
            "returncode": 0,
            "stdout": "abc123",
            "stderr": "",
        }
        self.assertEqual(command_stdout(payload, "repo.head"), "abc123")
        self.assertEqual(command_stdout("legacy123", "repo.head"), "legacy123")
        with self.assertRaisesRegex(ValueError, "repo.head"):
            command_stdout({**payload, "returncode": 1}, "repo.head")

    def test_serial_contract(self):
        self.write([
            event([request_id], [step], step)
            for request_id in ("a", "b")
            for step in CAPTURE_STEPS
        ])
        report = trace_contract(
            self.root, ["a", "b"], cap=1, expected="serial"
        )
        self.assertEqual(report["singleton_batch_count"], 98)
        self.assertEqual(report["same_age_batch_count"], 0)

    def test_homogeneous_contract_requires_all_49_paired_steps(self):
        rows = [event(["a", "b"], [step, step], step) for step in CAPTURE_STEPS]
        self.write(rows)
        report = trace_contract(
            self.root, ["a", "b"], cap=2, expected="homogeneous"
        )
        self.assertEqual(report["same_age_batch_count"], 49)
        self.assertEqual(report["mixed_age_batch_count"], 0)

        bad = copy.deepcopy(rows)
        bad[10] = event(["a", "b"], [10, 11], 10)
        bad[11] = event(["a", "b"], [11, 10], 11)
        self.write(bad)
        with self.assertRaisesRegex(ValueError, "same-step"):
            trace_contract(
                self.root, ["a", "b"], cap=2, expected="homogeneous"
            )

    def test_homogeneous_contract_allows_early_singletons(self):
        rows = [event(["a"], [0]), event(["b"], [0])]
        rows += [event(["a", "b"], [step, step], step) for step in range(1, 49)]
        self.write(rows)
        report = trace_contract(
            self.root, ["a", "b"], cap=2, expected="homogeneous"
        )
        self.assertEqual(report["same_age_batch_count"], 48)
        self.assertEqual(report["singleton_batch_count"], 2)

    def test_homogeneous_contract_rejects_too_few_paired_steps(self):
        first_paired = 49 - (MIN_HOMOGENEOUS_BATCHES - 1)
        rows = [
            event([request_id], [step], step)
            for request_id in ("a", "b")
            for step in range(first_paired)
        ]
        rows += [
            event(["a", "b"], [step, step], step)
            for step in range(first_paired, 49)
        ]
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "at least 32 same-step"):
            trace_contract(
                self.root, ["a", "b"], cap=2, expected="homogeneous"
            )
        report = trace_contract(
            self.root, ["a", "b"], cap=2, expected="homogeneous",
            strict_expectation=False,
        )
        self.assertFalse(report["expectation_satisfied"])
        self.assertEqual(report["same_age_batch_count"], MIN_HOMOGENEOUS_BATCHES - 1)
        self.assertIn("at least 32 same-step", report["expectation_violations"][0])

    def test_heterogeneous_contract(self):
        rows = [event(["a"], [step], step) for step in range(16)]
        rows += [
            event(["a", "b"], [step + 16, step], 100 + step)
            for step in range(33)
        ]
        rows += [event(["b"], [step], 200 + step) for step in range(33, 49)]
        self.write(rows)
        report = trace_contract(
            self.root, ["a", "b"], cap=2, expected="heterogeneous"
        )
        self.assertEqual(report["mixed_age_batch_count"], 33)
        self.assertEqual(report["steps_per_request"], {"a": 49, "b": 49})

    def test_warmup_does_not_satisfy_heterogeneous_contract(self):
        rows = [
            event([request_id], [step], step)
            for request_id in ("a", "b")
            for step in CAPTURE_STEPS
        ]
        rows.append(event(["warmup-a", "warmup-b"], [10, 0], 999))
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "no measured mixed-age"):
            trace_contract(
                self.root, ["a", "b"], cap=2, expected="heterogeneous"
            )

    def test_capture_step_cli_validation(self):
        with patch.object(sys, "argv", [
            "run_profile.py", "--cache-dir", str(self.root),
            "--output-dir", str(self.root / "out"), "--capture-latents",
            "--capture-latent-steps", "0,1,2,48",
        ]):
            args = run_profile.parse_args()
        self.assertEqual(args.capture_latent_steps, "0,1,2,48")

    def test_all_step_latent_comparison(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable locally; runs in the pinned Colab venv")

        roots = [self.root / name for name in ("serial", "candidate")]
        requests = [
            {"request_id": "serial-a", "request_index": 0, "seed": 17},
            {"request_id": "candidate-a", "request_index": 0, "seed": 17},
        ]
        for root, request in zip(roots, requests, strict=True):
            directory = root / "debug_latents" / request["request_id"]
            directory.mkdir(parents=True)
            for step in CAPTURE_STEPS:
                torch.save({
                    "request_id": request["request_id"],
                    "step_index": step,
                    "cfg_branch_policy": "drop_image",
                    "latents": torch.ones(2, 3),
                    "time_index": torch.full((2,), step + 1),
                }, directory / f"after_step_{step:02d}.pt")
        candidate_path = (
            roots[1] / "debug_latents" / requests[1]["request_id"]
            / "after_step_07.pt"
        )
        payload = torch.load(candidate_path, weights_only=True)
        payload["latents"][0, 0] = 2
        torch.save(payload, candidate_path)
        trace_steps = {
            step: {"batch_size": 2, "steps": [step, step]}
            for step in CAPTURE_STEPS
        }
        report = compare_trajectory(
            roots[0], roots[1], requests[0], requests[1], trace_steps
        )
        self.assertEqual(report["first_nonexact_step"], 7)
        self.assertEqual(len(report["steps"]), 49)


if __name__ == "__main__":
    unittest.main()
