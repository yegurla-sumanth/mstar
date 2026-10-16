"""CPU contracts for the bounded BAGEL operator-localization diagnostic."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/bagel/scripts"))

from analyze_operator_localization import (
    CAPTURE_STEPS,
    COMPUTE_STAGES,
    CONTROL_STAGES,
    STAGE_CAPTURE_STEPS,
    compare_arm,
    tensor_metrics,
    tensor_record,
    trace_steps,
)
import run_profile


def event(ids: list[str], ages: list[int], batch_id: int) -> dict:
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
    }


class OperatorLocalizationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_eight_step_trace_accepts_same_age_b2(self):
        trace_dir = self.root / "execution_trace"
        trace_dir.mkdir()
        rows = [event(["a", "b"], [step, step], step) for step in CAPTURE_STEPS]
        (trace_dir / "worker.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        report = trace_steps(self.root, ["a", "b"], cap=2)
        self.assertEqual(set(report["a"]), set(CAPTURE_STEPS))
        self.assertTrue(all(row["batch_size"] == 2 for row in report["a"].values()))

    def test_profile_cli_records_bounded_capture(self):
        with patch.object(sys, "argv", [
            "run_profile.py",
            "--cache-dir", str(self.root),
            "--output-dir", str(self.root / "out"),
            "--num-requests", "2",
            "--measured-seeds", "20261016,20261117",
            "--image-gen-max-iters", "8",
            "--capture-latents",
            "--capture-latent-steps", "0,1,2,3,4,5,6,7",
            "--capture-stages",
            "--capture-stage-steps", "0,1,2,3",
        ]):
            args = run_profile.parse_args()
        self.assertEqual(args.image_gen_max_iters, 8)
        self.assertEqual(args.capture_stage_steps, "0,1,2,3")
        self.assertEqual(args.measured_seeds, "20261016,20261117")

    def test_digest_is_exact_even_when_sample_matches(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable locally; runs in the pinned Colab venv")

        left = torch.zeros(5000)
        right = left.clone()
        # tensor_record keeps full values for this size; emulate a large-stage
        # record whose deterministic sample did not include the changed entry.
        left_record = tensor_record(left)
        right_record = tensor_record(right)
        left_record["values"] = right_record["values"] = None
        right_record["sha256"] = "different"
        metrics = tensor_metrics(left_record, right_record)
        self.assertFalse(metrics["exact"])
        self.assertEqual(metrics["exact_basis"], "sha256_full_tensor_bytes")

    def test_first_changed_internal_operation_is_reported(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable locally; runs in the pinned Colab venv")

        serial_root = self.root / "serial"
        candidate_root = self.root / "candidate"
        serial_request = {"request_id": "bagel-measured-00-serial", "seed": 20261016}
        candidate_request = {"request_id": "bagel-measured-00-batched", "seed": 20261016}
        serial_value = torch.zeros(2, 3)
        candidate_value = serial_value.clone()

        for step in CAPTURE_STEPS:
            serial_output = serial_value + 1
            candidate_output = candidate_value + 1
            if step == 1:
                candidate_output = candidate_output.clone()
                candidate_output[0, 0] += 0.25

            for root, request, input_value, output_value, candidate in (
                (serial_root, serial_request, serial_value, serial_output, False),
                (candidate_root, candidate_request, candidate_value, candidate_output, True),
            ):
                latent_dir = root / "debug_latents" / request["request_id"]
                latent_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "request_id": request["request_id"],
                    "step_index": step,
                    "cfg_branch_policy": "drop_image",
                    "latents": output_value,
                    "time_index": torch.full((2,), step + 1),
                }, latent_dir / f"after_step_{step:02d}.pt")

                if step in STAGE_CAPTURE_STEPS:
                    stages = {}
                    for name in CONTROL_STAGES + COMPUTE_STAGES[:-1]:
                        value = input_value
                        if name in {"output_latent_pre_split", "combined_velocity"}:
                            value = output_value
                        elif candidate and step == 1 and (
                            name.startswith("hidden_") or name.startswith("velocity_")
                        ):
                            value = output_value
                        stages[name] = tensor_record(value)
                    stage_dir = root / "debug_stages" / request["request_id"]
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "request_id": request["request_id"],
                        "step_index": step,
                        "capture_kind": "full_stage_fingerprints",
                        "batch_size": 2 if candidate else 1,
                        "batch_steps": [step, step] if candidate else [step],
                        "request_batch_index": 0,
                        "stages": stages,
                    }, stage_dir / f"step_{step:02d}.pt")

            serial_value = serial_output
            candidate_value = candidate_output

        trace = {
            step: {"batch_id": step, "batch_size": 2, "steps": [step, step]}
            for step in CAPTURE_STEPS
        }
        result = compare_arm(
            serial_root,
            candidate_root,
            serial_request,
            candidate_request,
            trace,
        )
        self.assertEqual(result["first_nonexact_output_step"], 1)
        self.assertTrue(result["input_exact_at_first_nonexact_output"])
        self.assertEqual(result["first_nonexact_stage_at_boundary"], "hidden_main")


if __name__ == "__main__":
    unittest.main()
