#!/usr/bin/env python3
"""Analyze BAGEL CFG branch pruning correctness and diagnostic timings."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from mstar.model.bagel.batching import cfg_branch_labels_for_step

from compare_batch_parity import compare_images
from smoke_image import atomic_json


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


def _step_cuda_by_request(run_dir: Path) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for row in _load_trace_rows(run_dir):
        if not (
            row.get("event_type") == "component_complete"
            and row.get("node_name") == "LLM"
            and row.get("graph_walk") == "image_gen"
            and row.get("batch_size") == 1
        ):
            continue
        request_ids = row.get("request_ids", [])
        progress = row.get("dynamic_loop_iter_counts", {})
        if len(request_ids) != 1 or len(progress) != 1:
            continue
        counts = next(iter(progress.values()))
        if "image_gen_loop" not in counts:
            continue
        duration = row.get("cuda_duration_ms")
        if duration is not None:
            result[request_ids[0]][int(counts["image_gen_loop"])] = float(
                duration
            )
    return result


def _latent_comparison(
    run_dir: Path, reference_id: str, candidate_id: str
) -> list[dict]:
    reference_dir = run_dir / "debug_latents" / reference_id
    candidate_dir = run_dir / "debug_latents" / candidate_id
    rows = []
    for reference_path in sorted(reference_dir.glob("after_step_*.pt")):
        candidate_path = candidate_dir / reference_path.name
        if not candidate_path.exists():
            raise ValueError(f"missing candidate latent dump: {candidate_path}")
        reference = torch.load(
            reference_path, map_location="cpu", weights_only=True
        )["latents"].float()
        candidate = torch.load(
            candidate_path, map_location="cpu", weights_only=True
        )["latents"].float()
        if reference.shape != candidate.shape:
            raise ValueError(
                f"latent shape mismatch for {reference_path.name}: "
                f"{tuple(reference.shape)} vs {tuple(candidate.shape)}"
            )
        if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
            raise ValueError(f"non-finite latent values in {reference_path.name}")
        delta = candidate - reference
        reference_l2 = torch.linalg.vector_norm(reference)
        rows.append({
            "step_index": int(reference_path.stem.rsplit("_", 1)[1]),
            "max_abs_error": float(delta.abs().max()),
            "mean_abs_error": float(delta.abs().mean()),
            "rmse": float(torch.sqrt(torch.mean(delta.square()))),
            "relative_l2_error": float(
                torch.linalg.vector_norm(delta) / (reference_l2 + 1e-12)
            ),
            "exact_fraction": float((delta == 0).float().mean()),
        })
    if not rows:
        raise ValueError(f"no reference latent dumps found in {reference_dir}")
    return rows


def _plot(payload: dict, output: Path) -> None:
    policies = payload["policies"]
    steps = list(range(49))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    for policy in policies:
        counts = [
            len(row["active_labels"])
            for row in payload["expected_branch_schedule"][policy]
        ]
        ax.step(steps, counts, where="mid", label=policy)
    ax.set_title("Physical CFG rows by refinement step")
    ax.set_xlabel("step index")
    ax.set_ylabel("virtual rows per request")
    ax.set_yticks([1, 2, 3])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    latency = [payload["policy_summary"][p]["median_client_seconds"] for p in policies]
    ax.bar(policies, latency)
    ax.set_title("Full request latency (diagnostic)")
    ax.set_ylabel("seconds")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    for policy in policies:
        by_step = payload["policy_summary"][policy]["median_cuda_ms_by_step"]
        xs = sorted(int(step) for step in by_step)
        ax.plot(xs, [by_step[str(step)] for step in xs], label=policy)
    ax.axvline(40.5, color="black", linestyle="--", alpha=0.5)
    ax.set_title("LLM CUDA time per refinement step")
    ax.set_xlabel("step index")
    ax.set_ylabel("CUDA ms")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    candidates = [p for p in policies if p != "reference"]
    for policy in candidates:
        rows = payload["comparisons"][policy]["latent_steps"]
        ax.plot(
            [row["step_index"] for row in rows],
            [max(row["relative_l2_error"], 1e-12) for row in rows],
            marker="o",
            label=policy,
        )
    ax.set_yscale("log")
    ax.set_title("Post-step latent error vs reference")
    ax.set_xlabel("step index")
    ax.set_ylabel("relative L2 error (log scale)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        "BAGEL CFG branch-pruning gate\n"
        "Latent dumps synchronize the GPU; timing is diagnostic only",
        fontsize=14,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary = json.loads((run_dir / "cfg_sweep_summary.json").read_text())
    if summary.get("status") != "success":
        raise ValueError("CFG sweep did not complete successfully")
    requests = summary["measured_requests"]
    workload = summary["workload"]
    policies = workload["policies"]
    by_repeat_policy = {
        (int(row["repeat"]), row["cfg_branch_policy"]): row
        for row in requests
    }
    expected_count = int(workload["repeats"]) * len(policies)
    if len(by_repeat_policy) != expected_count:
        raise ValueError(
            f"expected {expected_count} unique measured arms, found "
            f"{len(by_repeat_policy)}"
        )

    cuda = _step_cuda_by_request(run_dir)
    expected_schedule = {}
    for policy in policies:
        expected_schedule[policy] = [
            {
                "step_index": step,
                "active_labels": list(cfg_branch_labels_for_step(
                    step,
                    requires_cfg=True,
                    policy=policy,
                    num_timesteps=int(workload["num_timesteps"]),
                    timestep_shift=float(workload["timestep_shift"]),
                    cfg_text_scale=float(workload["cfg_text_scale"]),
                    cfg_img_scale=float(workload["cfg_img_scale"]),
                    cfg_interval=workload["cfg_interval"],
                )),
            }
            for step in range(int(workload["num_timesteps"]) - 1)
        ]

    policy_summary = {}
    for policy in policies:
        rows = [
            by_repeat_policy[(repeat, policy)]
            for repeat in range(int(workload["repeats"]))
        ]
        step_samples: dict[int, list[float]] = defaultdict(list)
        cuda_totals = []
        for row in rows:
            request_steps = cuda.get(row["request_id"], {})
            for step, duration in request_steps.items():
                step_samples[step].append(duration)
            cuda_totals.append(sum(request_steps.values()) / 1000.0)
        policy_summary[policy] = {
            "client_seconds": [row["client_elapsed_seconds"] for row in rows],
            "median_client_seconds": statistics.median(
                row["client_elapsed_seconds"] for row in rows
            ),
            "refinement_cuda_seconds": cuda_totals,
            "median_refinement_cuda_seconds": (
                statistics.median(cuda_totals) if cuda_totals else math.nan
            ),
            "traced_refinement_steps_per_repeat": [
                len(cuda.get(row["request_id"], {})) for row in rows
            ],
            "median_cuda_ms_by_step": {
                str(step): statistics.median(values)
                for step, values in sorted(step_samples.items())
            },
            "expected_virtual_rows": sum(
                len(row["active_labels"])
                for row in expected_schedule[policy]
            ),
        }

    reference_latency = policy_summary["reference"]["median_client_seconds"]
    reference_cuda = policy_summary["reference"][
        "median_refinement_cuda_seconds"
    ]
    for policy in policies:
        item = policy_summary[policy]
        item["client_speedup_vs_reference"] = (
            reference_latency / item["median_client_seconds"]
        )
        item["refinement_cuda_speedup_vs_reference"] = (
            reference_cuda / item["median_refinement_cuda_seconds"]
            if item["median_refinement_cuda_seconds"] > 0
            else math.nan
        )

    comparisons = {}
    for policy in policies:
        if policy == "reference":
            continue
        image_rows = []
        latent_rows = []
        for repeat in range(int(workload["repeats"])):
            reference = by_repeat_policy[(repeat, "reference")]
            candidate = by_repeat_policy[(repeat, policy)]
            image_rows.append(compare_images(reference, candidate))
            latent_rows.append(_latent_comparison(
                run_dir, reference["request_id"], candidate["request_id"]
            ))
        comparisons[policy] = {
            "images": image_rows,
            "latent_steps": latent_rows[0],
            "latent_steps_by_repeat": latent_rows,
            "minimum_image_psnr_db": min(row["psnr_db"] for row in image_rows),
            "maximum_image_abs_error": max(
                row["max_abs_pixel_error"] for row in image_rows
            ),
            "maximum_latent_relative_l2_error": max(
                row["relative_l2_error"]
                for repeat_rows in latent_rows
                for row in repeat_rows
            ),
        }

    payload = {
        "status": "complete",
        "measurement_boundary": (
            "Single loaded server; one warm-up; sequential B=1 policy arms. "
            "Intermediate latent dumps synchronize selected steps, so these "
            "timings diagnose potential only and are not performance claims."
        ),
        "policies": policies,
        "expected_branch_schedule": expected_schedule,
        "policy_summary": policy_summary,
        "comparisons": comparisons,
        "gate_notes": [
            "reference preserves the original three CFG rows at all 49 steps",
            "main_only is an intentionally unguided negative control",
            "no post-hoc numerical acceptance threshold is imposed",
            "repeat without debug latent dumps before reporting speedups",
        ],
    }
    output = args.output or run_dir / "cfg_branch_analysis.json"
    plot_path = run_dir / "cfg_branch_analysis.png"
    payload["plot"] = str(plot_path)
    _plot(payload, plot_path)
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
