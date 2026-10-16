#!/usr/bin/env python3
"""Render diagnostic plots from a staggered BAGEL profiling run."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def component_label(row: dict) -> str:
    node = row.get("node_name", "unknown")
    walk = row.get("graph_walk", "unknown")
    if node == "LLM" and walk == "prefill_text":
        return "Prompt prefill"
    if node == "LLM" and walk.startswith("image_gen"):
        return "LLM refinement"
    if node == "vae_decoder":
        return "VAE decode"
    if node == "vae_encoder":
        return "VAE encode"
    if node == "vit_encoder":
        return "ViT encode"
    return f"{node}/{walk}"


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Colab exports its notebook-only inline backend to child processes. The
    # isolated serving venv intentionally does not install Jupyter, so force a
    # headless backend before importing Matplotlib.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    summary = json.loads((input_dir / "profile_summary.json").read_text())
    if summary.get("status") != "success":
        raise RuntimeError("profile_summary.json is not a successful run")
    measured = sorted(summary["measured_requests"], key=lambda row: row["request_index"])
    if not measured or any(row.get("status") != "success" for row in measured):
        raise RuntimeError("measured request set is empty or contains failures")
    measured_ids = {row["request_id"] for row in measured}
    request_by_id = {row["request_id"]: row for row in measured}
    t0_perf = summary["measurement_start_perf_ns"]
    t0_wall = summary["measurement_start_wall_ns"]

    trace_rows = []
    for path in sorted((input_dir / "execution_trace").glob("*.jsonl")):
        trace_rows.extend(read_jsonl(path))
    if not trace_rows:
        raise RuntimeError("no execution trace rows found")
    components = [
        row
        for row in trace_rows
        if row.get("event_type") == "component_complete" and measured_ids.intersection(row.get("request_ids", []))
    ]
    scheduled = [
        row
        for row in trace_rows
        if row.get("event_type") == "batch_scheduled" and measured_ids.intersection(row.get("request_ids", []))
    ]
    if not components:
        raise RuntimeError("trace has no component completions for measured requests")

    colors = {
        "Prompt prefill": "#4C78A8",
        "LLM refinement": "#59A14F",
        "VAE decode": "#F28E2B",
        "VAE encode": "#E15759",
        "ViT encode": "#B07AA1",
    }
    fallback = "#9C755F"

    # Request/component Gantt.
    fig, ax = plt.subplots(figsize=(15, max(4.5, len(measured) * 0.8)))
    y_of = {row["request_id"]: len(measured) - 1 - i for i, row in enumerate(measured)}
    for request in measured:
        request_id = request["request_id"]
        y = y_of[request_id]
        submit = (request["submit_perf_ns"] - t0_perf) / 1e9
        complete = (request["complete_perf_ns"] - t0_perf) / 1e9
        ax.hlines(y, submit, complete, color="#D0D0D0", linewidth=8, zorder=0)
        ax.text(
            complete,
            y + 0.27,
            f"{request['client_elapsed_seconds']:.1f}s",
            fontsize=8,
            ha="right",
            va="bottom",
        )
    for row in components:
        start = (row["component_start_ns"] - t0_perf) / 1e9
        end = (row["gpu_complete_ns"] - t0_perf) / 1e9
        label = component_label(row)
        for request_id in measured_ids.intersection(row.get("request_ids", [])):
            ax.barh(
                y_of[request_id],
                max(end - start, 0),
                left=start,
                height=0.52,
                color=colors.get(label, fallback),
                alpha=0.9,
                zorder=2,
            )
    for row in scheduled:
        scheduled_s = (row["scheduled_ns"] - t0_perf) / 1e9
        handles = row.get("request_handles", [])
        ready = row.get("ready_observed_ns", {})
        for request_id, handle in zip(row.get("request_ids", []), handles, strict=False):
            if request_id not in measured_ids or str(handle) not in ready:
                continue
            ready_s = (ready[str(handle)] - t0_perf) / 1e9
            ax.hlines(
                y_of[request_id],
                ready_s,
                scheduled_s,
                color="#333333",
                linewidth=1.2,
                linestyles="dashed",
                zorder=3,
            )
    ax.set_yticks(
        [y_of[row["request_id"]] for row in measured],
        [f"R{row['request_index']}" for row in measured],
    )
    ax.set_xlabel("Seconds from measured-workload start")
    ax.set_ylabel("Request")
    ax.set_title("BAGEL request and component timeline")
    ax.grid(axis="x", alpha=0.2)
    legend_labels = sorted({component_label(row) for row in components})
    handles = [Patch(facecolor=colors.get(label, fallback), label=label) for label in legend_labels]
    handles.append(plt.Line2D([0], [0], color="#333333", linestyle="dashed", label="Ready → scheduled"))
    ax.legend(handles=handles, loc="upper left", ncols=2, fontsize=9)
    fig.tight_layout()
    request_plot = output_dir / "request_component_timeline.png"
    fig.savefig(request_plot, dpi=180)
    plt.close(fig)

    # Scheduler/component lanes. The pale bar is CPU start through observed GPU
    # completion; the saturated bar is the CUDA-event duration anchored at the
    # completion point for alignment.
    lane_labels = sorted({component_label(row) for row in components})
    lane_of = {label: len(lane_labels) - 1 - i for i, label in enumerate(lane_labels)}
    fig, ax = plt.subplots(figsize=(15, max(4.0, len(lane_labels) * 1.0)))
    for row in sorted(components, key=lambda item: item["component_start_ns"]):
        label = component_label(row)
        y = lane_of[label]
        start = (row["component_start_ns"] - t0_perf) / 1e9
        end = (row["gpu_complete_ns"] - t0_perf) / 1e9
        color = colors.get(label, fallback)
        ax.barh(y, max(end - start, 0), left=start, height=0.58, color=color, alpha=0.25)
        cuda_ms = row.get("cuda_duration_ms")
        if cuda_ms is not None:
            cuda_start = end - cuda_ms / 1000
            ax.barh(
                y,
                cuda_ms / 1000,
                left=cuda_start,
                height=0.34,
                color=color,
                alpha=0.95,
            )
        if row.get("batch_size", 1) > 1:
            ax.text(
                (start + end) / 2,
                y,
                f"B={row['batch_size']}",
                fontsize=7,
                ha="center",
                va="center",
            )
    ax.set_yticks([lane_of[label] for label in lane_labels], lane_labels)
    ax.set_xlabel("Seconds from measured-workload start")
    ax.set_ylabel("Component lane")
    ax.set_title("M* component scheduling and CUDA activity")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(
        handles=[
            Patch(facecolor="#777777", alpha=0.25, label="Component service window"),
            Patch(facecolor="#777777", alpha=0.95, label="CUDA-event duration"),
        ],
        loc="upper left",
        fontsize=9,
    )
    fig.tight_layout()
    scheduler_plot = output_dir / "scheduler_component_timeline.png"
    fig.savefig(scheduler_plot, dpi=180)
    plt.close(fig)

    # GPU telemetry aligned by wall time.
    gpu_samples = json.loads((input_dir / "gpu_samples.json").read_text())["samples"]
    valid_samples = [row for row in gpu_samples if "utilization_gpu_percent" in row and "memory_used_mib" in row]
    if not valid_samples:
        raise RuntimeError("gpu_samples.json has no valid nvidia-smi samples")
    xs = [(row["sample_time_ns"] - t0_wall) / 1e9 for row in valid_samples]
    util = [row["utilization_gpu_percent"] for row in valid_samples]
    memory_gib = [row["memory_used_mib"] / 1024 for row in valid_samples]
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    axes[0].plot(xs, util, color="#4C78A8", linewidth=1.2)
    axes[0].fill_between(xs, util, color="#4C78A8", alpha=0.18)
    axes[0].set_ylabel("GPU utilization (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.2)
    axes[1].plot(xs, memory_gib, color="#F28E2B", linewidth=1.2)
    axes[1].set_ylabel("GPU memory used (GiB)")
    axes[1].set_xlabel("Seconds from measured-workload start")
    axes[1].grid(alpha=0.2)
    fig.suptitle("GPU telemetry aligned with the measured workload")
    fig.tight_layout()
    telemetry_plot = output_dir / "gpu_telemetry.png"
    fig.savefig(telemetry_plot, dpi=180)
    plt.close(fig)

    per_request_counts = {}
    warnings = []
    expected = {
        "LLM/prefill_text": 1,
        "LLM/image_gen": 49,
        "vae_decoder/image_gen": 1,
    }
    for request_id in sorted(measured_ids, key=lambda rid: request_by_id[rid]["request_index"]):
        counts = Counter()
        for row in components:
            if request_id in row.get("request_ids", []):
                counts[f"{row['node_name']}/{row['graph_walk']}"] += 1
        per_request_counts[request_id] = dict(sorted(counts.items()))
        for key, expected_count in expected.items():
            if counts[key] != expected_count:
                warnings.append(f"{request_id}: expected {expected_count} {key} executions, traced {counts[key]}")

    manifest = {
        "status": "complete" if not warnings else "complete_with_warnings",
        "input_dir": str(input_dir),
        "plots": [str(request_plot), str(scheduler_plot), str(telemetry_plot)],
        "measured_request_count": len(measured),
        "component_completion_count": len(components),
        "scheduler_ready_batch_count": sum(1 for row in scheduled if row.get("ready_observed_ns")),
        "per_request_component_counts": per_request_counts,
        "expected_per_request_component_counts": expected,
        "warnings": warnings,
        "interpretation_boundary": (
            "Diagnostic trace only. Component service windows include host-side "
            "preparation and synchronization; CUDA-event durations are device "
            "intervals. This six-request run validates instrumentation and does "
            "not establish throughput or tail-latency conclusions."
        ),
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
