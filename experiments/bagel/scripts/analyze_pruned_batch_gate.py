#!/usr/bin/env python3
"""Validate pruned B=1/B=2 work and report numerical drift, not a speedup."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_smoke import atomic_json

CAPTURE_STEPS = (0, 20, 40, 41, 48)


def trace_contract(root: Path, request_ids: list[str], cap: int) -> dict:
    """Check every measured step; warm-ups cannot satisfy mixed-age evidence."""
    steps = {request_id: {} for request_id in request_ids}
    if not steps or len(steps) != len(request_ids):
        raise ValueError("Empty or duplicate request IDs")
    mixed = []
    cuda_ms = 0.0
    for path in sorted((root / "execution_trace").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if (row.get("event_type"), row.get("node_name"), row.get("graph_walk")) != (
                "component_complete", "LLM", "image_gen"
            ):
                continue
            ids = row.get("request_ids", [])
            if not set(ids).intersection(steps):
                continue
            handles = row.get("request_handles", [])
            if (not set(ids).issubset(steps) or len(ids) != len(set(ids))
                    or len(ids) != len(handles) or len(ids) != row.get("batch_size")
                    or not 1 <= len(ids) <= cap):
                raise ValueError("Unexpected membership/size in measured image batch")
            metadata = row.get("model_metadata", {})
            ages = []
            for request_id, handle in zip(ids, handles):
                handle = str(handle)
                age = row["dynamic_loop_iter_counts"][handle]["image_gen_loop"]
                labels = metadata["cfg_active_labels_by_request_handle"][handle]
                virtual_rows = metadata["cfg_virtual_rows_by_request_handle"][handle]
                if labels != ["main", "cfg_text"] or virtual_rows != 2:
                    raise ValueError(f"{request_id} step {age}: expected two drop_image rows")
                if age in steps[request_id]:
                    raise ValueError(f"Duplicate step {age} for {request_id}")
                steps[request_id][age] = virtual_rows
                ages.append(age)
            cuda_ms += float(row["cuda_duration_ms"])
            if len(set(ages)) > 1:
                mixed.append({"batch_id": row.get("batch_id"), "request_ids": ids, "steps": ages})
    for request_id, seen in steps.items():
        if set(seen) != set(range(49)):
            raise ValueError(f"{request_id}: expected exactly steps 0..48; found {sorted(seen)}")
    if cap > 1 and not mixed:
        raise ValueError("No measured heterogeneous-step image batch; warm-ups do not count")
    return {
        "steps_per_request": {key: len(value) for key, value in steps.items()},
        "virtual_rows_per_request": {key: sum(value.values()) for key, value in steps.items()},
        "heterogeneous_batch_count": len(mixed), "examples": mixed[:5],
        "refinement_cuda_seconds_diagnostic_only": cuda_ms / 1000,
    }


def compare_latents(serial_root, batched_root, reference, candidate):
    import torch

    rows = []
    for step in CAPTURE_STEPS:
        payloads = []
        for root, request in ((serial_root, reference), (batched_root, candidate)):
            path = root / "debug_latents" / request["request_id"] / f"after_step_{step:02d}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["step_index"] != step or payload["request_id"] != request["request_id"]:
                raise ValueError(f"Latent identity mismatch: {path}")
            if payload["cfg_branch_policy"] != "drop_image":
                raise ValueError(f"Unexpected latent policy: {path}")
            latent = payload["latents"]
            if not torch.isfinite(latent).all():
                raise ValueError(f"Nonfinite latent: {path}")
            payloads.append(latent)
        left, right = payloads
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Latent shape/dtype mismatch at step {step}")
        delta = right.double() - left.double()
        rows.append({
            "step": step, "exact": bool(torch.equal(left, right)),
            "max_abs_error": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()),
            "relative_l2": float(delta.norm() / left.double().norm().clamp_min(1e-12)),
        })
    return rows


def analyze(serial_root: Path, batched_root: Path) -> dict:
    from compare_batch_parity import compare_images, load_summary, measured_by_index

    summaries = [load_summary(root) for root in (serial_root, batched_root)]
    for key in ("prompt", "base_seed", "width", "height", "measured_requests", "warmup_requests",
                "stagger_seconds", "cfg_branch_policy", "cfg_text_scale", "cfg_img_scale", "cfg_interval"):
        if summaries[0]["workload"][key] != summaries[1]["workload"][key]:
            raise ValueError(f"Workload mismatch: {key}")
    if summaries[0]["workload"]["cfg_branch_policy"] != "drop_image":
        raise ValueError("Both arms must use drop_image")
    runs = [json.loads((root / "run_summary.json").read_text()) for root in (serial_root, batched_root)]
    for run in runs:
        if run["status"] != "success" or run["shutdown"]["method"] != "sigterm":
            raise ValueError("Both owned servers must have completed and shut down cleanly")
        if not run["workload"]["latent_capture_enabled"] or not run["workload"]["torch_compile_disabled"]:
            raise ValueError("This gate requires eager execution and latent capture")
    if runs[0]["config"]["sha256"] != runs[1]["config"]["sha256"]:
        raise ValueError("Configuration hashes differ")
    if runs[0]["repo"]["head"] != runs[1]["repo"]["head"]:
        raise ValueError("Code revisions differ")
    left, right = [measured_by_index(summary) for summary in summaries]
    if set(left) != set(right) or set(left) != set(range(summaries[0]["workload"]["measured_requests"])):
        raise ValueError("Missing/mismatched measured request indexes")
    evidence = [trace_contract(root, [row["request_id"] for row in records.values()], cap)
                for root, records, cap in ((serial_root, left, 1), (batched_root, right, 2))]
    comparisons = []
    for index in sorted(left):
        for root, record in ((serial_root, left[index]), (batched_root, right[index])):
            # Use the artifact within the supplied folder, even after download.
            path = root / "requests" / record["request_id"] / "image.png"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record["image"]["sha256"]:
                raise ValueError(f"PNG checksum mismatch: {path}")
            record["image"]["path"] = str(path)
        comparison = compare_images(left[index], right[index])
        comparison["png_bytes_equal"] = (Path(left[index]["image"]["path"]).read_bytes()
                                         == Path(right[index]["image"]["path"]).read_bytes())
        comparison["latents"] = compare_latents(serial_root, batched_root, left[index], right[index])
        comparisons.append(comparison)
    exact = all(row["png_bytes_equal"] and all(step["exact"] for step in row["latents"])
                for row in comparisons)
    return {
        "status": "exact_at_checked_boundaries" if exact else "numerical_review_required",
        "performance_authorized": False,
        "workload": summaries[0]["workload"],
        "serial_trace": evidence[0], "batched_trace": evidence[1], "requests": comparisons,
        "interpretation": "Checks cover five stored latent boundaries and final images, not every tensor. "
                          "Non-exact B=2 results require numerical review; no tolerance is silently chosen. "
                          "Latent dumps synchronize the GPU, so this is not a performance comparison.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-dir", type=Path, required=True)
    parser.add_argument("--batched-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.serial_dir, args.batched_dir)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
