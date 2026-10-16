#!/usr/bin/env python3
"""Analyze the paired eager BAGEL CFG branch-pruning benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from mstar.model.bagel.batching import cfg_branch_labels_for_step
from smoke_image import atomic_json


POLICIES = ("reference", "drop_image")
EXPECTED_STEPS = tuple(range(49))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_trace_rows(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "execution_trace").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _image_steps_by_request(run_dir: Path) -> dict[str, dict[int, dict]]:
    result: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in _load_trace_rows(run_dir):
        if not (
            row.get("event_type") == "component_complete"
            and row.get("node_name") == "LLM"
            and row.get("graph_walk") == "image_gen"
            and row.get("batch_size") == 1
        ):
            continue
        request_ids = row.get("request_ids", [])
        request_handles = row.get("request_handles", [])
        progress = row.get("dynamic_loop_iter_counts", {})
        if len(request_ids) != 1 or len(request_handles) != 1 or len(progress) != 1:
            continue
        counts = next(iter(progress.values()))
        if "image_gen_loop" not in counts:
            continue
        metadata = row.get("model_metadata", {})
        handle = str(request_handles[0])
        labels = metadata.get(
            "cfg_active_labels_by_request_handle", {}
        ).get(handle)
        virtual_rows = metadata.get(
            "cfg_virtual_rows_by_request_handle", {}
        ).get(handle)
        if labels is None or virtual_rows is None:
            raise ValueError(
                f"trace row for {request_ids[0]} lacks prepared CFG metadata"
            )
        step = int(counts["image_gen_loop"])
        if step in result[request_ids[0]]:
            raise ValueError(f"duplicate image step {step} for {request_ids[0]}")
        result[request_ids[0]][step] = {
            "step_index": step,
            "active_labels": labels,
            "virtual_rows": int(virtual_rows),
            "cuda_duration_ms": float(row["cuda_duration_ms"]),
        }
    return result


def _distribution(values: list[float]) -> dict:
    if not values:
        return {
            "values": [],
            "count": 0,
            "mean": None,
            "median": None,
            "sample_stddev": None,
            "minimum": None,
            "maximum": None,
            "q1": None,
            "q3": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in values],
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "sample_stddev": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
    }


def _request_gpu_samples(samples: list[dict], request: dict) -> dict:
    selected = [
        row for row in samples
        if request["submit_wall_ns"] <= row.get("sample_time_ns", -1)
        <= request["complete_wall_ns"]
        and "memory_used_mib" in row
        and "utilization_gpu_percent" in row
    ]
    memory = [row["memory_used_mib"] for row in selected]
    utilization = [row["utilization_gpu_percent"] for row in selected]
    return {
        "sample_count": len(selected),
        "peak_memory_used_mib": max(memory) if memory else None,
        "median_utilization_percent": (
            statistics.median(utilization) if utilization else None
        ),
    }


def _plot(payload: dict, output: Path) -> None:
    repeats = list(range(payload["repeats"]))
    colors = {"reference": "#4C78A8", "drop_image": "#F58518"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    for policy in POLICIES:
        schedule = payload["observed_cfg_schedule"][policy]
        ax.step(
            [row["step_index"] for row in schedule],
            [row["virtual_rows"] for row in schedule],
            where="mid", label=policy, color=colors[policy],
        )
    ax.set_title("Prepared CFG rows by refinement step")
    ax.set_xlabel("step index")
    ax.set_ylabel("virtual rows")
    ax.set_yticks([1, 2, 3])
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for policy in POLICIES:
        values = payload["policy_summary"][policy]["client_seconds"]["values"]
        ax.plot(repeats, values, marker="o", label=policy, color=colors[policy])
    ax.set_title("Paired full-request latency")
    ax.set_xlabel("pair index")
    ax.set_ylabel("seconds")
    ax.set_xticks(repeats)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    for policy in POLICIES:
        values = payload["policy_summary"][policy][
            "refinement_cuda_seconds"
        ]["values"]
        ax.plot(repeats, values, marker="o", label=policy, color=colors[policy])
    ax.set_title("Traced LLM refinement CUDA time")
    ax.set_xlabel("pair index")
    ax.set_ylabel("seconds")
    ax.set_xticks(repeats)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    speedups = payload["paired_comparison"]["client_speedup"]["values"]
    cuda_speedups = payload["paired_comparison"][
        "refinement_cuda_speedup"
    ]["values"]
    width = 0.36
    x = np.arange(len(repeats))
    ax.bar(x - width / 2, speedups, width, label="client")
    ax.bar(x + width / 2, cuda_speedups, width, label="refinement CUDA")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_title("Paired speedup: reference / drop_image")
    ax.set_xlabel("pair index")
    ax.set_ylabel("×")
    ax.set_xticks(x, repeats)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle(
        "BAGEL exact CFG image-branch pruning — eager B=1\n"
        "Two warm-ups, alternating paired order, no latent capture",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary = json.loads((run_dir / "cfg_benchmark_summary.json").read_text())
    if summary.get("status") != "success":
        raise ValueError("CFG performance benchmark did not complete successfully")
    workload = summary["workload"]
    if tuple(workload["policies"]) != POLICIES:
        raise ValueError(f"expected policies {POLICIES}, got {workload['policies']}")
    repeats = int(workload["repeats"])
    requests = summary["measured_requests"]
    by_pair_policy = {
        (int(row["repeat"]), row["cfg_branch_policy"]): row for row in requests
    }
    if len(by_pair_policy) != repeats * len(POLICIES):
        raise ValueError("measured request matrix is incomplete or duplicated")

    trace = _image_steps_by_request(run_dir)
    gpu_samples = json.loads((run_dir / "gpu_samples.json").read_text())[
        "samples"
    ]
    expected_schedule = {
        policy: [
            list(cfg_branch_labels_for_step(
                step,
                requires_cfg=True,
                policy=policy,
                num_timesteps=int(workload["num_timesteps"]),
                timestep_shift=float(workload["timestep_shift"]),
                cfg_text_scale=float(workload["cfg_text_scale"]),
                cfg_img_scale=float(workload["cfg_img_scale"]),
                cfg_interval=workload["cfg_interval"],
            ))
            for step in EXPECTED_STEPS
        ]
        for policy in POLICIES
    }

    policy_summary = {}
    observed_cfg_schedule = {}
    for policy in POLICIES:
        policy_requests = [by_pair_policy[(repeat, policy)] for repeat in range(repeats)]
        client_values = []
        cuda_values = []
        memory_peaks = []
        utilization_medians = []
        schedules = []
        for request in policy_requests:
            request_steps = trace.get(request["request_id"], {})
            if tuple(sorted(request_steps)) != EXPECTED_STEPS:
                raise ValueError(
                    f"{request['request_id']} has {len(request_steps)} traced "
                    "refinement steps instead of 49"
                )
            schedule = [request_steps[step] for step in EXPECTED_STEPS]
            observed = [row["active_labels"] for row in schedule]
            if observed != expected_schedule[policy]:
                raise ValueError(
                    f"prepared CFG schedule mismatch for {request['request_id']}"
                )
            schedules.append(schedule)
            client_values.append(float(request["client_elapsed_seconds"]))
            cuda_values.append(
                sum(row["cuda_duration_ms"] for row in schedule) / 1000.0
            )
            sampled = _request_gpu_samples(gpu_samples, request)
            if sampled["peak_memory_used_mib"] is not None:
                memory_peaks.append(sampled["peak_memory_used_mib"])
            if sampled["median_utilization_percent"] is not None:
                utilization_medians.append(sampled["median_utilization_percent"])
        first_signature = [
            {
                "step_index": row["step_index"],
                "active_labels": row["active_labels"],
                "virtual_rows": row["virtual_rows"],
            }
            for row in schedules[0]
        ]
        if any(
            [row["active_labels"] for row in schedule]
            != [row["active_labels"] for row in schedules[0]]
            for schedule in schedules[1:]
        ):
            raise ValueError(f"{policy} prepared CFG schedule changed across repeats")
        observed_cfg_schedule[policy] = first_signature
        policy_summary[policy] = {
            "client_seconds": _distribution(client_values),
            "refinement_cuda_seconds": _distribution(cuda_values),
            "observed_virtual_rows_per_request": sum(
                row["virtual_rows"] for row in first_signature
            ),
            "peak_sampled_gpu_memory_mib": _distribution(memory_peaks),
            "median_sampled_gpu_utilization_percent": _distribution(
                utilization_medians
            ),
        }

    exact_pairs = []
    client_speedups = []
    cuda_speedups = []
    client_savings = []
    cuda_savings = []
    for repeat in range(repeats):
        reference = by_pair_policy[(repeat, "reference")]
        pruned = by_pair_policy[(repeat, "drop_image")]
        exact = reference["image"]["sha256"] == pruned["image"]["sha256"]
        exact_pairs.append(exact)
        ref_client = float(reference["client_elapsed_seconds"])
        drop_client = float(pruned["client_elapsed_seconds"])
        ref_cuda = sum(
            row["cuda_duration_ms"]
            for row in trace[reference["request_id"]].values()
        ) / 1000.0
        drop_cuda = sum(
            row["cuda_duration_ms"]
            for row in trace[pruned["request_id"]].values()
        ) / 1000.0
        client_speedups.append(ref_client / drop_client)
        cuda_speedups.append(ref_cuda / drop_cuda)
        client_savings.append(ref_client - drop_client)
        cuda_savings.append(ref_cuda - drop_cuda)
    if not all(exact_pairs):
        raise ValueError("at least one paired final PNG was not exactly equal")

    payload = {
        "status": "complete",
        "scope": "eager B=1 exact drop_image optimization",
        "repeats": repeats,
        "measurement_boundary": summary["measurement_boundary"],
        "policy_summary": policy_summary,
        "observed_cfg_schedule": observed_cfg_schedule,
        "paired_comparison": {
            "exact_png_pairs": exact_pairs,
            "exact_png_pair_count": sum(exact_pairs),
            "client_speedup": _distribution(client_speedups),
            "refinement_cuda_speedup": _distribution(cuda_speedups),
            "client_seconds_saved": _distribution(client_savings),
            "refinement_cuda_seconds_saved": _distribution(cuda_savings),
        },
        "limitations": [
            "one prompt at five seeds on one A100 runtime",
            "eager image refinement only; compiled CFG pruning remains unresolved",
            "B=1 sequential requests; this does not measure continuous batching",
            "nvidia-smi memory/utilization samples are diagnostic, not allocator accounting",
            "five pairs support a pilot estimate, not a tail-latency claim",
        ],
    }
    output = args.output or run_dir / "cfg_branch_benchmark_analysis.json"
    plot = run_dir / "cfg_branch_benchmark_analysis.png"
    payload["plot"] = str(plot)
    _plot(payload, plot)
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
