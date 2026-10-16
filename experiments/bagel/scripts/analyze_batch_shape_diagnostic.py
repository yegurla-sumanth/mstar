#!/usr/bin/env python3
"""Attribute BAGEL drift to B=2 shape or mixed refinement ages.

This is a correctness diagnostic.  Every saved latent is loaded on CPU and
compared with the same prompt/seed request from the serial B=1 arm.  Timing is
reported only as diagnostic metadata because latent capture synchronizes CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from run_smoke import atomic_json

CAPTURE_STEPS = tuple(range(49))


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


def trace_contract(
    root: Path,
    request_ids: list[str],
    *,
    cap: int,
    expected: str,
) -> dict[str, Any]:
    """Validate all measured transitions and classify their actual batching."""
    if expected not in {"serial", "homogeneous", "heterogeneous"}:
        raise ValueError(f"Unknown trace expectation: {expected}")
    if not request_ids or len(request_ids) != len(set(request_ids)):
        raise ValueError("Measured request IDs must be nonempty and unique")

    measured = set(request_ids)
    per_request: dict[str, dict[int, dict[str, Any]]] = {
        request_id: {} for request_id in request_ids
    }
    same_age_batches = []
    mixed_age_batches = []
    singleton_batches = 0
    cuda_ms = 0.0

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
            raise ValueError(f"Unexpected measured image batch membership: {ids}")

        metadata = row.get("model_metadata", {})
        ages = []
        for request_id, handle_value in zip(ids, handles, strict=True):
            handle = str(handle_value)
            try:
                age = int(row["dynamic_loop_iter_counts"][handle]["image_gen_loop"])
                labels = metadata["cfg_active_labels_by_request_handle"][handle]
                virtual_rows = metadata["cfg_virtual_rows_by_request_handle"][handle]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Incomplete trace metadata for {request_id}") from exc
            if labels != ["main", "cfg_text"] or virtual_rows != 2:
                raise ValueError(f"{request_id} step {age}: expected two drop_image rows")
            if age in per_request[request_id]:
                raise ValueError(f"Duplicate step {age} for {request_id}")
            ages.append(age)
            per_request[request_id][age] = {
                "batch_id": row.get("batch_id"),
                "batch_size": len(ids),
                "request_ids": list(ids),
                "steps": None,
            }

        for request_id, age in zip(ids, ages, strict=True):
            per_request[request_id][age]["steps"] = list(ages)
        cuda_ms += float(row.get("cuda_duration_ms", 0.0))
        if len(ids) == 1:
            singleton_batches += 1
        else:
            evidence = {
                "batch_id": row.get("batch_id"),
                "request_ids": list(ids),
                "steps": list(ages),
            }
            if len(set(ages)) == 1:
                same_age_batches.append(evidence)
            else:
                mixed_age_batches.append(evidence)

    for request_id, seen in per_request.items():
        if set(seen) != set(CAPTURE_STEPS):
            raise ValueError(
                f"{request_id}: expected exactly steps 0..48; found {sorted(seen)}"
            )

    if expected == "serial" and (same_age_batches or mixed_age_batches):
        raise ValueError("Serial B=1 arm contains a measured multi-request batch")
    if expected == "homogeneous" and (
        len(same_age_batches) != len(CAPTURE_STEPS)
        or mixed_age_batches
        or singleton_batches
    ):
        raise ValueError(
            "Homogeneous B=2 arm did not keep both measured requests in exactly "
            "49 same-step batches"
        )
    if expected == "heterogeneous" and not mixed_age_batches:
        raise ValueError("Heterogeneous B=2 arm has no measured mixed-age batch")

    return {
        "expected": expected,
        "steps_per_request": {
            request_id: len(steps) for request_id, steps in per_request.items()
        },
        "virtual_rows_per_request": {
            request_id: 2 * len(steps) for request_id, steps in per_request.items()
        },
        "singleton_batch_count": singleton_batches,
        "same_age_batch_count": len(same_age_batches),
        "mixed_age_batch_count": len(mixed_age_batches),
        "same_age_examples": same_age_batches[:5],
        "mixed_age_examples": mixed_age_batches[:5],
        "refinement_cuda_seconds_diagnostic_only": cuda_ms / 1000.0,
        "per_request_steps": per_request,
    }


def _load_latent(root: Path, request: dict[str, Any], step: int):
    import torch

    path = root / "debug_latents" / request["request_id"] / f"after_step_{step:02d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("request_id") != request["request_id"] or payload.get("step_index") != step:
        raise ValueError(f"Latent identity mismatch: {path}")
    if payload.get("cfg_branch_policy") != "drop_image":
        raise ValueError(f"Unexpected latent policy: {path}")
    latent = payload["latents"]
    time_index = payload["time_index"]
    if not torch.isfinite(latent).all() or not torch.isfinite(time_index).all():
        raise ValueError(f"Nonfinite latent payload: {path}")
    return latent, time_index


def compare_trajectory(
    serial_root: Path,
    candidate_root: Path,
    serial_request: dict[str, Any],
    candidate_request: dict[str, Any],
    trace_steps: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    import torch

    rows = []
    for step in CAPTURE_STEPS:
        left, left_time = _load_latent(serial_root, serial_request, step)
        right, right_time = _load_latent(candidate_root, candidate_request, step)
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Latent shape/dtype mismatch at step {step}")
        if not torch.equal(left_time, right_time):
            raise ValueError(f"Time-index mismatch at step {step}")
        delta = right.double() - left.double()
        execution = trace_steps[step]
        rows.append({
            "step": step,
            "exact": bool(torch.equal(left, right)),
            "max_abs_error": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()),
            "relative_l2": float(
                delta.norm() / left.double().norm().clamp_min(1e-12)
            ),
            "candidate_batch_size": execution["batch_size"],
            "candidate_batch_steps": execution["steps"],
        })
    divergent = [row for row in rows if not row["exact"]]
    return {
        "request_index": int(serial_request["request_index"]),
        "seed": int(serial_request["seed"]),
        "first_nonexact_step": divergent[0]["step"] if divergent else None,
        "final_relative_l2": rows[-1]["relative_l2"],
        "maximum_relative_l2": max(row["relative_l2"] for row in rows),
        "steps": rows,
    }


def _validated_runs(roots: dict[str, Path]) -> tuple[dict, dict, dict]:
    from compare_batch_parity import load_summary, measured_by_index

    summaries = {name: load_summary(root) for name, root in roots.items()}
    runs = {
        name: json.loads((root / "run_summary.json").read_text())
        for name, root in roots.items()
    }
    shared_keys = (
        "prompt", "base_seed", "width", "height", "measured_requests",
        "warmup_requests", "cfg_branch_policy", "cfg_text_scale",
        "cfg_img_scale", "cfg_interval",
    )
    serial_workload = summaries["serial_b1"]["workload"]
    for name, summary in summaries.items():
        for key in shared_keys:
            if summary["workload"][key] != serial_workload[key]:
                raise ValueError(f"Workload mismatch for {name}: {key}")
        if summary["workload"]["cfg_branch_policy"] != "drop_image":
            raise ValueError(f"{name} did not use drop_image")
    expected_staggers = {
        "serial_b1": 8.0,
        "homogeneous_b2": 0.0,
        "heterogeneous_b2": 8.0,
    }
    for name, expected in expected_staggers.items():
        if float(summaries[name]["workload"]["stagger_seconds"]) != expected:
            raise ValueError(f"Unexpected stagger for {name}")
    heads = {run["repo"]["head"] for run in runs.values()}
    configs = {run["config"]["sha256"] for run in runs.values()}
    for name, run in runs.items():
        if run.get("status") != "success" or run.get("shutdown", {}).get("method") != "sigterm":
            raise ValueError(f"{name} did not complete with clean owned shutdown")
        workload = run["workload"]
        if not workload.get("latent_capture_enabled") or not workload.get("torch_compile_disabled"):
            raise ValueError(f"{name} must use eager execution and latent capture")
        if workload.get("latent_capture_steps") != list(CAPTURE_STEPS):
            raise ValueError(f"{name} did not capture every step 0..48")
    if len(heads) != 1 or len(configs) != 1:
        raise ValueError("Arms used different code revisions or model configurations")
    measured = {name: measured_by_index(summary) for name, summary in summaries.items()}
    expected_indexes = set(range(int(serial_workload["measured_requests"])))
    if any(set(rows) != expected_indexes for rows in measured.values()):
        raise ValueError("Arms contain missing or mismatched measured request indexes")
    return summaries, runs, measured


def analyze(roots: dict[str, Path]) -> dict[str, Any]:
    from compare_batch_parity import compare_images

    summaries, runs, measured = _validated_runs(roots)
    traces = {
        name: trace_contract(
            roots[name],
            [row["request_id"] for row in measured[name].values()],
            cap=1 if name == "serial_b1" else 2,
            expected={
                "serial_b1": "serial",
                "homogeneous_b2": "homogeneous",
                "heterogeneous_b2": "heterogeneous",
            }[name],
        )
        for name in roots
    }

    comparisons: dict[str, list[dict[str, Any]]] = {}
    images: dict[str, list[dict[str, Any]]] = {}
    for candidate_name in ("homogeneous_b2", "heterogeneous_b2"):
        trajectory_rows = []
        image_rows = []
        for index in sorted(measured["serial_b1"]):
            reference = measured["serial_b1"][index]
            candidate = measured[candidate_name][index]
            for root, record in (
                (roots["serial_b1"], reference),
                (roots[candidate_name], candidate),
            ):
                image_path = root / "requests" / record["request_id"] / "image.png"
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if digest != record["image"]["sha256"]:
                    raise ValueError(f"PNG checksum mismatch: {image_path}")
                record["image"]["path"] = str(image_path)
            image = compare_images(reference, candidate)
            image["png_bytes_equal"] = (
                Path(reference["image"]["path"]).read_bytes()
                == Path(candidate["image"]["path"]).read_bytes()
            )
            image_rows.append(image)
            trajectory_rows.append(compare_trajectory(
                roots["serial_b1"], roots[candidate_name], reference, candidate,
                traces[candidate_name]["per_request_steps"][candidate["request_id"]],
            ))
        comparisons[candidate_name] = trajectory_rows
        images[candidate_name] = image_rows

    homogeneous_nonexact = any(
        row["first_nonexact_step"] is not None
        for row in comparisons["homogeneous_b2"]
    )
    heterogeneous_nonexact = any(
        row["first_nonexact_step"] is not None
        for row in comparisons["heterogeneous_b2"]
    )
    if homogeneous_nonexact and heterogeneous_nonexact:
        assessment = "batching_general_candidate"
    elif not homogeneous_nonexact and heterogeneous_nonexact:
        assessment = "mixed_age_specific_candidate"
    elif homogeneous_nonexact and not heterogeneous_nonexact:
        assessment = "unexpected_homogeneous_only_drift"
    else:
        assessment = "exact_in_both_b2_arms"

    # The detailed per-step membership is embedded in trajectory rows. Avoid
    # duplicating it under the trace summary and inflating the report.
    trace_summaries = {}
    for name, trace in traces.items():
        trace_summaries[name] = {
            key: value for key, value in trace.items()
            if key != "per_request_steps"
        }

    return {
        "status": "diagnostic_complete",
        "performance_authorized": False,
        "hypothesis_assessment": assessment,
        "workload": summaries["serial_b1"]["workload"],
        "code_revision": next(iter({run["repo"]["head"] for run in runs.values()})),
        "trace_contracts": trace_summaries,
        "latent_comparisons": comparisons,
        "image_comparisons": images,
        "interpretation": (
            "All 49 saved post-step latents were compared by request index and seed. "
            "A homogeneous-arm difference shows that B=2 shape alone can change the "
            "trajectory; a difference unique to the heterogeneous arm points to an "
            "additional mixed-age effect. No numerical acceptance threshold is imposed. "
            "All timing is diagnostic because latent capture synchronizes CUDA."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-dir", type=Path, required=True)
    parser.add_argument("--homogeneous-dir", type=Path, required=True)
    parser.add_argument("--heterogeneous-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "serial_b1": args.serial_dir.resolve(),
        "homogeneous_b2": args.homogeneous_dir.resolve(),
        "heterogeneous_b2": args.heterogeneous_dir.resolve(),
    }
    report = analyze(roots)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
