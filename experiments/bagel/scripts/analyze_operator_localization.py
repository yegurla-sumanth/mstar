#!/usr/bin/env python3
"""Localize the first BAGEL B=1/B=2 numerical difference.

The diagnostic keeps the checkpoint's 50-boundary timestep schedule but runs
only the first eight Euler iterations. It compares one fixed target seed in a
serial arm, a repeated B=2 arm, and a B=2 arm whose partner seed differs. Internal tensors are
request-local before comparison, and every comparison is gated on trace and
state-handoff integrity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_batch_shape_diagnostic import command_stdout
from run_smoke import atomic_json

TARGET_SEED = 20261016
MAX_ITERS = 8
CAPTURE_STEPS = tuple(range(MAX_ITERS))
STAGE_CAPTURE_STEPS = tuple(range(4))
ARMS = (
    "serial_b1",
    "batched_partner_a",
    "batched_partner_a_repeat",
    "batched_partner_b",
)
CONTROL_STAGES = (
    "input_time_index",
    "output_time_index",
    "shifted_timestep",
    "euler_dt",
    "vae_position_ids",
    "local_text_indexes",
    "local_vae_token_indexes",
)
COMPUTE_STAGES = (
    "input_latent",
    "vae_projection",
    "timestep_embedding",
    "position_embedding",
    "projected_latent",
    "transformer_input_latent_tokens",
    "transformer_input_sequence",
    "hidden_main",
    "hidden_cfg_text",
    "velocity_main",
    "velocity_cfg_text",
    "combined_velocity",
    "output_latent_pre_split",
    "output_latent_post_split",
)


def load_summary(root: Path) -> dict[str, Any]:
    path = root / "profile_summary.json"
    summary = json.loads(path.read_text())
    if summary.get("status") != "success":
        raise ValueError(f"Profile is not successful: {path}")
    return summary


def measured_by_index(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {}
    for row in summary.get("measured_requests", []):
        if row.get("status") != "success":
            raise ValueError(
                f"Request {row.get('request_index')} is not successful"
            )
        index = int(row["request_index"])
        if index in rows:
            raise ValueError(f"Duplicate measured request index {index}")
        rows[index] = row
    return rows


def _component_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "execution_trace").glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid trace JSON at {path}:{number}") from exc
            if (row.get("event_type"), row.get("node_name"), row.get("graph_walk")) == (
                "component_complete", "LLM", "image_gen"
            ):
                rows.append(row)
    return rows


def trace_steps(root: Path, request_ids: list[str], cap: int) -> dict[str, Any]:
    measured = set(request_ids)
    if not measured or len(measured) != len(request_ids):
        raise ValueError("Measured request IDs must be nonempty and unique")
    per_request = {request_id: {} for request_id in request_ids}
    for row in _component_rows(root):
        ids = row.get("request_ids", [])
        if not measured.intersection(ids):
            continue
        handles = row.get("request_handles", [])
        if (
            not set(ids).issubset(measured)
            or len(ids) != len(set(ids))
            or len(ids) != len(handles)
            or len(ids) != row.get("batch_size")
            or not 1 <= len(ids) <= cap
        ):
            raise ValueError(f"Unexpected measured image batch: {ids}")
        ages = []
        for request_id, handle_value in zip(ids, handles, strict=True):
            handle = str(handle_value)
            try:
                age = int(row["dynamic_loop_iter_counts"][handle]["image_gen_loop"])
                labels = row["model_metadata"][
                    "cfg_active_labels_by_request_handle"
                ][handle]
                virtual_rows = row["model_metadata"][
                    "cfg_virtual_rows_by_request_handle"
                ][handle]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Incomplete trace metadata for {request_id}") from exc
            if labels != ["main", "cfg_text"] or virtual_rows != 2:
                raise ValueError(
                    f"{request_id} step {age}: expected two drop_image rows"
                )
            if age in per_request[request_id]:
                raise ValueError(f"Duplicate step {age} for {request_id}")
            ages.append(age)
        for request_id, age in zip(ids, ages, strict=True):
            per_request[request_id][age] = {
                "batch_id": row.get("batch_id"),
                "batch_size": len(ids),
                "steps": list(ages),
            }
    for request_id, seen in per_request.items():
        if set(seen) != set(CAPTURE_STEPS):
            raise ValueError(
                f"{request_id}: expected steps 0..{MAX_ITERS - 1}; "
                f"found {sorted(seen)}"
            )
    return per_request


def _load_latent(root: Path, request: dict[str, Any], step: int):
    import torch

    path = (
        root / "debug_latents" / request["request_id"]
        / f"after_step_{step:02d}.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("request_id") != request["request_id"]
        or payload.get("step_index") != step
    ):
        raise ValueError(f"Latent identity mismatch: {path}")
    if payload.get("cfg_branch_policy") != "drop_image":
        raise ValueError(f"Unexpected latent policy: {path}")
    for name in ("latents", "time_index"):
        if not torch.isfinite(payload[name]).all():
            raise ValueError(f"Nonfinite {name}: {path}")
    return payload


def _load_stage(root: Path, request: dict[str, Any], step: int) -> dict[str, Any]:
    import torch

    path = root / "debug_stages" / request["request_id"] / f"step_{step:02d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("request_id") != request["request_id"]
        or payload.get("step_index") != step
        or payload.get("capture_kind") != "full_stage_fingerprints"
    ):
        raise ValueError(f"Stage identity mismatch: {path}")
    stages = payload.get("stages", {})
    expected = set(CONTROL_STAGES + COMPUTE_STAGES[:-1])
    if set(stages) != expected:
        raise ValueError(
            f"Unexpected stage set at {path}: "
            f"missing={sorted(expected - set(stages))}, "
            f"extra={sorted(set(stages) - expected)}"
        )
    for name, record in stages.items():
        if not record.get("finite"):
            raise ValueError(f"Nonfinite stage {name}: {path}")
    return payload


def _load_partner_probe(
    root: Path, request: dict[str, Any]
) -> dict[str, Any]:
    import torch

    path = root / "debug_stages" / request["request_id"] / "step_00.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("request_id") != request["request_id"]
        or payload.get("step_index") != 0
        or payload.get("capture_kind") != "partner_input_probe"
        or set(payload.get("stages", {})) != {"input_latent"}
    ):
        raise ValueError(f"Partner probe identity mismatch: {path}")
    record = payload["stages"]["input_latent"]
    if not record.get("finite"):
        raise ValueError(f"Nonfinite partner input: {path}")
    return record


def tensor_record(tensor) -> dict[str, Any]:
    """Make an in-memory comparison record for a small saved latent."""
    import hashlib
    import torch

    cpu = tensor.detach().contiguous().cpu()
    flat = cpu.reshape(-1)
    numeric = flat.double()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "numel": flat.numel(),
        "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "finite": bool(torch.isfinite(cpu).all()),
        "mean": float(numeric.mean()) if flat.numel() else 0.0,
        "l2_norm": float(numeric.norm()) if flat.numel() else 0.0,
        "max_abs": float(numeric.abs().max()) if flat.numel() else 0.0,
        "sample_indexes": torch.arange(flat.numel(), dtype=torch.int64),
        "sample_values": flat,
        "values": cpu,
    }


def tensor_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    import torch

    result = {
        "left_shape": left["shape"],
        "right_shape": right["shape"],
        "left_dtype": left["dtype"],
        "right_dtype": right["dtype"],
    }
    if left["shape"] != right["shape"] or left["dtype"] != right["dtype"]:
        return {**result, "comparable": False, "exact": False}
    exact = left["sha256"] == right["sha256"]
    left_values = left.get("values")
    right_values = right.get("values")
    if left_values is not None and right_values is not None:
        left_numeric = left_values.double()
        delta = right_values.double() - left_numeric
        basis = "full_tensor"
    else:
        if not torch.equal(left["sample_indexes"], right["sample_indexes"]):
            raise ValueError("Stage fingerprints used different sample indexes")
        left_numeric = left["sample_values"].double()
        delta = right["sample_values"].double() - left_numeric
        basis = "deterministic_sample"
    return {
        **result,
        "comparable": True,
        "exact": exact,
        "exact_basis": "sha256_full_tensor_bytes",
        "error_metric_basis": basis,
        "max_abs_error": float(delta.abs().max()) if delta.numel() else 0.0,
        "rmse": float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
        "relative_l2": float(
            delta.norm() / left_numeric.norm().clamp_min(1e-12)
        ),
    }


def _target_by_seed(rows: dict[int, dict], seed: int) -> dict[str, Any]:
    matches = [row for row in rows.values() if int(row["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected one request with seed {seed}; found {len(matches)}")
    return matches[0]


def _validated_runs(roots: dict[str, Path]):
    if set(roots) != set(ARMS):
        raise ValueError(f"Expected arms {ARMS}; found {sorted(roots)}")
    summaries = {name: load_summary(root) for name, root in roots.items()}
    runs = {
        name: json.loads((root / "run_summary.json").read_text())
        for name, root in roots.items()
    }
    shared_keys = (
        "prompt", "width", "height", "cfg_branch_policy",
        "cfg_text_scale", "cfg_img_scale", "cfg_interval",
    )
    baseline = summaries["serial_b1"]["workload"]
    for name, summary in summaries.items():
        for key in shared_keys:
            if summary["workload"][key] != baseline[key]:
                raise ValueError(f"Workload mismatch for {name}: {key}")
        if summary["workload"]["cfg_branch_policy"] != "drop_image":
            raise ValueError(f"{name} did not use drop_image")

    heads = {
        command_stdout(run["repo"]["head"], f"{name}.repo.head")
        for name, run in runs.items()
    }
    configs = {run["config"]["sha256"] for run in runs.values()}
    for name, run in runs.items():
        if (
            run.get("status") != "success"
            or run.get("shutdown", {}).get("method") != "sigterm"
        ):
            raise ValueError(f"{name} did not complete with clean owned shutdown")
        workload = run["workload"]
        if workload.get("image_gen_max_iters") != MAX_ITERS:
            raise ValueError(f"{name} did not use {MAX_ITERS} diagnostic steps")
        if workload.get("configured_num_timesteps") != 50:
            raise ValueError(f"{name} changed the original timestep schedule")
        if not workload.get("torch_compile_disabled"):
            raise ValueError(f"{name} must use eager execution")
        if not workload.get("latent_capture_enabled") or not workload.get(
            "stage_capture_enabled"
        ):
            raise ValueError(f"{name} is missing diagnostic capture")
        if workload.get("latent_capture_steps") != list(CAPTURE_STEPS):
            raise ValueError(f"{name} latent-capture steps do not match")
        if workload.get("stage_capture_steps") != list(STAGE_CAPTURE_STEPS):
            raise ValueError(f"{name} stage-capture steps do not match")
        if workload.get("stage_capture_request_prefix") != "bagel-measured-00-":
            raise ValueError(f"{name} stage-capture target does not match")
    if len(heads) != 1 or len(configs) != 1:
        raise ValueError("Arms used different code revisions or configurations")

    measured = {name: measured_by_index(summary) for name, summary in summaries.items()}
    if len(measured["serial_b1"]) != 1:
        raise ValueError("Serial arm must contain one measured request")
    if any(len(measured[name]) != 2 for name in ARMS[1:]):
        raise ValueError("Each B=2 arm must contain two measured requests")
    targets = {
        name: _target_by_seed(rows, TARGET_SEED)
        for name, rows in measured.items()
    }
    partners = {
        name: next(
            row for row in measured[name].values()
            if row["request_id"] != targets[name]["request_id"]
        )
        for name in ARMS[1:]
    }
    if (
        partners["batched_partner_a"]["seed"]
        != partners["batched_partner_a_repeat"]["seed"]
    ):
        raise ValueError("Repeat arm did not retain the same partner seed")
    if partners["batched_partner_a"]["seed"] == partners["batched_partner_b"]["seed"]:
        raise ValueError("Partner-seed perturbation was not configured")
    traces = {
        name: trace_steps(
            roots[name],
            [row["request_id"] for row in measured[name].values()],
            cap=1 if name == "serial_b1" else 2,
        )
        for name in ARMS
    }
    return (
        summaries, runs, measured, targets, partners, traces,
        next(iter(heads)),
    )


def compare_arm(
    serial_root: Path,
    candidate_root: Path,
    serial_request: dict[str, Any],
    candidate_request: dict[str, Any],
    candidate_trace: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    import torch

    rows = []
    previous_serial_output = None
    previous_candidate_output = None
    for step in CAPTURE_STEPS:
        serial_post = _load_latent(serial_root, serial_request, step)
        candidate_post = _load_latent(candidate_root, candidate_request, step)
        serial_output = tensor_record(serial_post["latents"])
        candidate_output = tensor_record(candidate_post["latents"])
        row = {
            "step": step,
            "candidate_execution": candidate_trace[step],
            "stages": {
                "output_latent_post_split": tensor_metrics(
                    serial_output, candidate_output
                )
            },
        }
        if step in STAGE_CAPTURE_STEPS:
            serial_stage = _load_stage(serial_root, serial_request, step)
            candidate_stage = _load_stage(candidate_root, candidate_request, step)
            serial_tensors = dict(serial_stage["stages"])
            candidate_tensors = dict(candidate_stage["stages"])
            serial_tensors["output_latent_post_split"] = serial_output
            candidate_tensors["output_latent_post_split"] = candidate_output
            if not tensor_metrics(
                serial_tensors["output_latent_pre_split"], serial_output
            )["exact"]:
                raise ValueError(f"Serial split changed the output at step {step}")
            if not tensor_metrics(
                candidate_tensors["output_latent_pre_split"], candidate_output
            )["exact"]:
                raise ValueError(f"Candidate split changed the output at step {step}")
            if previous_serial_output is not None and not tensor_metrics(
                serial_tensors["input_latent"], previous_serial_output
            )["exact"]:
                raise ValueError(f"Serial state handoff changed at step {step}")
            if previous_candidate_output is not None and not tensor_metrics(
                candidate_tensors["input_latent"], previous_candidate_output
            )["exact"]:
                raise ValueError(f"Candidate state handoff changed at step {step}")
            row.update(
                candidate_payload_batch_size=int(candidate_stage["batch_size"]),
                candidate_payload_batch_steps=list(candidate_stage["batch_steps"]),
                candidate_request_batch_index=int(
                    candidate_stage["request_batch_index"]
                ),
                stages={
                    name: tensor_metrics(
                        serial_tensors[name], candidate_tensors[name]
                    )
                    for name in CONTROL_STAGES + COMPUTE_STAGES
                },
            )
        previous_serial_output = serial_output
        previous_candidate_output = candidate_output
        rows.append(row)

    first_output = next(
        (
            row for row in rows
            if not row["stages"]["output_latent_post_split"]["exact"]
        ),
        None,
    )
    first_stage = None
    input_exact_at_boundary = None
    controls_exact_at_boundary = None
    if first_output is not None:
        boundary_stages = first_output["stages"]
        if "input_latent" in boundary_stages:
            input_exact_at_boundary = boundary_stages["input_latent"]["exact"]
            controls_exact_at_boundary = all(
                boundary_stages[name]["exact"] for name in CONTROL_STAGES
            )
            first_stage = next(
                (
                    name for name in COMPUTE_STAGES[1:]
                    if not boundary_stages[name]["exact"]
                ),
                None,
            )
    return {
        "target_seed": int(candidate_request["seed"]),
        "first_nonexact_output_step": (
            first_output["step"] if first_output is not None else None
        ),
        "first_nonexact_output_was_b2": (
            first_output is None
            or first_output["candidate_execution"]["batch_size"] == 2
        ),
        "input_exact_at_first_nonexact_output": input_exact_at_boundary,
        "controls_exact_at_first_nonexact_output": controls_exact_at_boundary,
        "first_nonexact_output_inside_stage_window": (
            first_output is None or first_output["step"] in STAGE_CAPTURE_STEPS
        ),
        "first_nonexact_stage_at_boundary": first_stage,
        "steps": rows,
    }


def compare_batched_pair(
    roots: dict[str, Path],
    targets: dict[str, dict[str, Any]],
    partners: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left_target, right_target = targets[left_name], targets[right_name]
    left_partner, right_partner = partners[left_name], partners[right_name]
    partner_left_input = _load_partner_probe(roots[left_name], left_partner)
    partner_right_input = _load_partner_probe(roots[right_name], right_partner)
    partner_input_comparison = tensor_metrics(
        partner_left_input, partner_right_input
    )
    perturbation_realized = not partner_input_comparison["exact"]

    rows = []
    schedule_equal = True
    all_target_stages_exact = True
    for step in CAPTURE_STEPS:
        left_trace = traces[left_name][left_target["request_id"]][step]
        right_trace = traces[right_name][right_target["request_id"]][step]
        signature_equal = (
            left_trace["batch_size"] == right_trace["batch_size"]
            and left_trace["steps"] == right_trace["steps"]
        )
        schedule_equal &= signature_equal
        left_output = tensor_record(_load_latent(
            roots[left_name], left_target, step
        )["latents"])
        right_output = tensor_record(_load_latent(
            roots[right_name], right_target, step
        )["latents"])
        comparisons = {
            "output_latent_post_split": tensor_metrics(
                left_output, right_output
            )
        }
        payload_signature_equal = True
        if step in STAGE_CAPTURE_STEPS:
            left_payload = _load_stage(roots[left_name], left_target, step)
            right_payload = _load_stage(roots[right_name], right_target, step)
            left_stages = dict(left_payload["stages"])
            right_stages = dict(right_payload["stages"])
            left_stages["output_latent_post_split"] = left_output
            right_stages["output_latent_post_split"] = right_output
            comparisons = {
                name: tensor_metrics(left_stages[name], right_stages[name])
                for name in CONTROL_STAGES + COMPUTE_STAGES
            }
            payload_signature_equal = (
                left_payload["batch_size"] == right_payload["batch_size"]
                and left_payload["batch_steps"] == right_payload["batch_steps"]
                and left_payload["request_batch_index"]
                == right_payload["request_batch_index"]
            )
            signature_equal &= payload_signature_equal
        all_target_stages_exact &= all(
            row["exact"] for row in comparisons.values()
        )
        rows.append({
            "step": step,
            "schedule_equal": signature_equal,
            "payload_signature_equal": payload_signature_equal,
            "stages": comparisons,
        })
    return {
        "arms": [left_name, right_name],
        "partner_seeds": [int(left_partner["seed"]), int(right_partner["seed"])],
        "partner_inputs_differ": perturbation_realized,
        "partner_input_comparison": partner_input_comparison,
        "target_schedule_equal": schedule_equal,
        "all_target_stages_exact": all_target_stages_exact,
        "steps": rows,
    }


def _classification(stage: str | None) -> str:
    if stage is None:
        return "no_output_divergence"
    if stage == "vae_projection":
        return "vae_to_llm_projection_candidate"
    if stage == "timestep_embedding":
        return "timestep_embedding_candidate"
    if stage == "position_embedding":
        return "position_embedding_candidate"
    if stage == "projected_latent":
        return "embedding_add_candidate"
    if stage in {"transformer_input_latent_tokens", "transformer_input_sequence"}:
        return "request_packing_or_latent_store_candidate"
    if stage.startswith("hidden_"):
        return "transformer_batch_shape_candidate"
    if stage.startswith("velocity_"):
        return "velocity_projection_candidate"
    if stage == "combined_velocity":
        return "cfg_combination_candidate"
    if stage == "output_latent_pre_split":
        return "euler_update_candidate"
    if stage == "output_latent_post_split":
        return "output_split_candidate"
    return "unclassified_stage_candidate"


def analyze(roots: dict[str, Path]) -> dict[str, Any]:
    (
        summaries, _runs, _measured, targets, partners, traces, code_revision,
    ) = _validated_runs(roots)
    comparisons = {
        name: compare_arm(
            roots["serial_b1"], roots[name],
            targets["serial_b1"], targets[name],
            traces[name][targets[name]["request_id"]],
        )
        for name in ARMS[1:]
    }
    repeatability = compare_batched_pair(
        roots, targets, partners, traces,
        "batched_partner_a", "batched_partner_a_repeat",
    )
    partner_isolation = compare_batched_pair(
        roots, targets, partners, traces,
        "batched_partner_a", "batched_partner_b",
    )
    b2_steps = {
        name: [
            step for step, row in traces[name][targets[name]["request_id"]].items()
            if row["batch_size"] == 2
        ]
        for name in ARMS[1:]
    }
    schedules_realized = all(len(steps) >= 2 for steps in b2_steps.values())
    stage_windows_valid = all(
        row["first_nonexact_output_inside_stage_window"]
        for row in comparisons.values()
    )
    divergence_boundaries_were_b2 = all(
        row["first_nonexact_output_was_b2"] for row in comparisons.values()
    )
    boundaries_valid = all(
        row["input_exact_at_first_nonexact_output"] is not False
        and row["controls_exact_at_first_nonexact_output"] is not False
        for row in comparisons.values()
    )
    stages = {
        row["first_nonexact_stage_at_boundary"]
        for row in comparisons.values()
    }
    if not schedules_realized:
        status = "diagnostic_inconclusive"
        assessment = "insufficient_b2_steps"
    elif not stage_windows_valid:
        status = "diagnostic_inconclusive"
        assessment = "first_divergence_outside_stage_capture_window"
    elif not divergence_boundaries_were_b2:
        status = "diagnostic_inconclusive"
        assessment = "first_divergence_did_not_occur_in_b2_execution"
    elif not boundaries_valid:
        status = "diagnostic_inconclusive"
        assessment = "nonidentical_boundary_input"
    elif repeatability["partner_inputs_differ"]:
        status = "diagnostic_inconclusive"
        assessment = "repeat_arm_input_mismatch"
    elif not repeatability["target_schedule_equal"]:
        status = "diagnostic_inconclusive"
        assessment = "repeat_arms_used_different_schedules"
    elif not repeatability["all_target_stages_exact"]:
        status = "diagnostic_inconclusive"
        assessment = "run_to_run_nondeterminism_detected"
    elif not partner_isolation["partner_inputs_differ"]:
        status = "diagnostic_inconclusive"
        assessment = "partner_perturbation_not_realized"
    elif not partner_isolation["target_schedule_equal"]:
        status = "diagnostic_inconclusive"
        assessment = "partner_arms_used_different_schedules"
    elif not partner_isolation["all_target_stages_exact"]:
        status = "diagnostic_complete"
        assessment = "partner_data_coupling_candidate"
    elif len(stages) != 1:
        status = "diagnostic_complete"
        assessment = "localization_not_replicated"
    else:
        status = "diagnostic_complete"
        assessment = _classification(next(iter(stages)))

    trace_summary = {
        name: {
            "target_request_id": targets[name]["request_id"],
            "target_seed": int(targets[name]["seed"]),
            "target_steps": traces[name][targets[name]["request_id"]],
        }
        for name in ARMS
    }
    return {
        "status": status,
        "performance_authorized": False,
        "hypothesis_assessment": assessment,
        "code_revision": code_revision,
        "configured_num_timesteps": 50,
        "executed_refinement_steps": MAX_ITERS,
        "target_seed": TARGET_SEED,
        "b2_target_steps": b2_steps,
        "trace_summary": trace_summary,
        "serial_vs_batched": comparisons,
        "b2_repeatability": repeatability,
        "partner_isolation": partner_isolation,
        "workloads": {
            name: summary["workload"] for name, summary in summaries.items()
        },
        "interpretation": (
            "The graph stopped after eight Euler iterations but retained the "
            "checkpoint's original 50-boundary timestep schedule. Internal "
            "stage capture synchronizes CUDA, so no timing is a performance "
            "measurement. A localization is valid only when the target input "
            "is exact at the first nonexact output boundary."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-dir", type=Path, required=True)
    parser.add_argument("--partner-a-dir", type=Path, required=True)
    parser.add_argument("--partner-a-repeat-dir", type=Path, required=True)
    parser.add_argument("--partner-b-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "serial_b1": args.serial_dir.resolve(),
        "batched_partner_a": args.partner_a_dir.resolve(),
        "batched_partner_a_repeat": args.partner_a_repeat_dir.resolve(),
        "batched_partner_b": args.partner_b_dir.resolve(),
    }
    report = analyze(roots)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
