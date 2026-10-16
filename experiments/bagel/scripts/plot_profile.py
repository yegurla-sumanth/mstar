#!/usr/bin/env python3
"""Render diagnostic plots from a staggered BAGEL profiling run."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
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

    # The image-generation walk is scheduler-grouped but cannot execute as one
    # model batch (LLMSubmodule.can_batch excludes image_gen). Show whether its
    # device time scales with the number of grouped requests, alongside the two
    # complete ready-wait series that occur once per request.
    refinement_cuda_by_batch_size = defaultdict(list)
    for row in components:
        if component_label(row) != "LLM refinement":
            continue
        cuda_ms = row.get("cuda_duration_ms")
        if cuda_ms is not None:
            refinement_cuda_by_batch_size[int(row.get("batch_size", 1))].append(
                cuda_ms / 1000
            )

    wait_by_request = {
        request_id: {"Prompt prefill": [], "VAE decode": []}
        for request_id in measured_ids
    }
    for row in scheduled:
        label = component_label(row)
        if label not in ("Prompt prefill", "VAE decode"):
            continue
        ready_wait_ms = row.get("ready_wait_ms", {})
        for request_id, handle in zip(
            row.get("request_ids", []), row.get("request_handles", []), strict=False
        ):
            value = ready_wait_ms.get(str(handle))
            if request_id in measured_ids and value is not None:
                wait_by_request[request_id][label].append(value / 1000)

    diagnostic_plot = None
    if refinement_cuda_by_batch_size:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

        batch_sizes = sorted(refinement_cuda_by_batch_size)
        medians = [
            statistics.median(refinement_cuda_by_batch_size[size])
            for size in batch_sizes
        ]
        for size in batch_sizes:
            values = refinement_cuda_by_batch_size[size]
            axes[0].scatter(
                [size] * len(values),
                values,
                color=colors["LLM refinement"],
                alpha=0.28,
                s=24,
                edgecolors="none",
            )
        axes[0].plot(
            batch_sizes,
            medians,
            color=colors["LLM refinement"],
            marker="o",
            linewidth=2,
            label="Observed median",
        )
        if 1 in refinement_cuda_by_batch_size:
            single_median = statistics.median(refinement_cuda_by_batch_size[1])
            axes[0].plot(
                batch_sizes,
                [single_median * size for size in batch_sizes],
                color="#555555",
                linestyle="--",
                linewidth=1.4,
                label="B × median(B=1)",
            )
        for size, median in zip(batch_sizes, medians, strict=True):
            axes[0].annotate(
                f"n={len(refinement_cuda_by_batch_size[size])}",
                (size, median),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axes[0].set_xlabel("Scheduler group size B")
        axes[0].set_ylabel("CUDA duration per scheduler group (seconds)")
        axes[0].set_title("LLM refinement group cost")
        axes[0].set_xticks(batch_sizes)
        axes[0].grid(alpha=0.2)
        axes[0].legend(fontsize=9)

        ordered_requests = sorted(measured, key=lambda row: row["request_index"])
        x_positions = list(range(len(ordered_requests)))
        width = 0.36
        prefill_waits = []
        vae_waits = []
        for request in ordered_requests:
            waits = wait_by_request[request["request_id"]]
            prefill_waits.append(sum(waits["Prompt prefill"]))
            vae_waits.append(sum(waits["VAE decode"]))
        axes[1].bar(
            [x - width / 2 for x in x_positions],
            prefill_waits,
            width,
            color=colors["Prompt prefill"],
            label="Prompt prefill",
        )
        axes[1].bar(
            [x + width / 2 for x in x_positions],
            vae_waits,
            width,
            color=colors["VAE decode"],
            label="VAE decode",
        )
        axes[1].set_xticks(
            x_positions,
            [f"R{request['request_index']}" for request in ordered_requests],
        )
        axes[1].set_xlabel("Request")
        axes[1].set_ylabel("Ready-to-scheduled wait (seconds)")
        axes[1].set_title("Wait before short boundary components")
        axes[1].grid(axis="y", alpha=0.2)
        axes[1].legend(fontsize=9)

        fig.suptitle("Scheduler grouping cost and request-boundary waiting")
        fig.tight_layout()
        diagnostic_plot = output_dir / "scheduler_grouping_diagnostics.png"
        fig.savefig(diagnostic_plot, dpi=180)
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
                warnings.append(
                    f"{request_id}: expected {expected_count} {key} executions, "
                    f"traced {counts[key]}"
                )
        for label in ("Prompt prefill", "VAE decode"):
            observed = len(wait_by_request[request_id][label])
            if observed != 1:
                warnings.append(
                    f"{request_id}: expected one ready wait for {label}, traced {observed}"
                )

    refinement_batch_stats = {
        str(size): {
            "batch_count": len(values),
            "median_cuda_seconds": statistics.median(values),
            "total_cuda_seconds": sum(values),
        }
        for size, values in sorted(refinement_cuda_by_batch_size.items())
    }
    complete_waits = {
        label: [
            sum(wait_by_request[request["request_id"]][label])
            for request in sorted(measured, key=lambda row: row["request_index"])
        ]
        for label in ("Prompt prefill", "VAE decode")
    }
    cuda_seconds_by_component = defaultdict(float)
    cuda_intervals = []
    logical_component_memberships = 0
    for row in components:
        cuda_ms = row.get("cuda_duration_ms")
        if cuda_ms is None:
            continue
        duration_ns = cuda_ms * 1e6
        end_ns = row["gpu_complete_ns"]
        cuda_seconds_by_component[component_label(row)] += cuda_ms / 1000
        cuda_intervals.append((end_ns - duration_ns, end_ns))
        logical_component_memberships += len(measured_ids.intersection(row.get("request_ids", [])))
    cuda_intervals.sort()
    positive_gap_ns = 0.0
    apparent_overlap_ns = 0.0
    for previous, current in zip(cuda_intervals, cuda_intervals[1:], strict=False):
        delta_ns = current[0] - previous[1]
        if delta_ns >= 0:
            positive_gap_ns += delta_ns
        else:
            apparent_overlap_ns -= delta_ns

    measurement_seconds = (
        summary["measurement_finished_perf_ns"] - summary["measurement_start_perf_ns"]
    ) / 1e9
    measured_gpu_samples = [
        row
        for row in valid_samples
        if 0 <= (row["sample_time_ns"] - t0_wall) / 1e9 <= measurement_seconds
    ]
    if not measured_gpu_samples:
        raise RuntimeError("gpu_samples.json has no samples inside the measured window")
    utilization = [row["utilization_gpu_percent"] for row in measured_gpu_samples]
    memory_mib = [row["memory_used_mib"] for row in measured_gpu_samples]
    request_latencies = [row["client_elapsed_seconds"] for row in measured]
    request_makespan_seconds = (
        max(row["complete_perf_ns"] for row in measured)
        - min(row["submit_perf_ns"] for row in measured)
    ) / 1e9
    total_cuda_seconds = sum(cuda_seconds_by_component.values())
    measured_metrics = {
        "request_latencies_seconds": request_latencies,
        "request_makespan_seconds": request_makespan_seconds,
        "component_cuda_seconds": dict(sorted(cuda_seconds_by_component.items())),
        "component_cuda_share": {
            label: seconds / total_cuda_seconds
            for label, seconds in sorted(cuda_seconds_by_component.items())
        },
        "total_component_cuda_seconds": total_cuda_seconds,
        "estimated_cuda_positive_gap_seconds": positive_gap_ns / 1e9,
        "estimated_cuda_apparent_overlap_seconds": apparent_overlap_ns / 1e9,
        "logical_component_memberships": logical_component_memberships,
        "average_memberships_per_component_batch": (
            logical_component_memberships / len(components)
        ),
        "gpu_sample_count_during_measurement": len(measured_gpu_samples),
        "gpu_utilization_mean_percent": statistics.mean(utilization),
        "gpu_utilization_median_percent": statistics.median(utilization),
        "gpu_utilization_samples_at_least_95_percent_fraction": (
            sum(value >= 95 for value in utilization) / len(utilization)
        ),
        "gpu_memory_peak_mib": max(memory_mib),
        "gpu_memory_min_mib": min(memory_mib),
    }
    plot_paths = [request_plot, scheduler_plot, telemetry_plot]
    if diagnostic_plot is not None:
        plot_paths.append(diagnostic_plot)
    manifest = {
        "status": "complete" if not warnings else "complete_with_warnings",
        "input_dir": str(input_dir),
        "plots": [str(path) for path in plot_paths],
        "measured_request_count": len(measured),
        "component_completion_count": len(components),
        "scheduler_ready_batch_count": sum(
            1 for row in scheduled if row.get("ready_observed_ns")
        ),
        "refinement_cuda_by_scheduler_group_size": refinement_batch_stats,
        "complete_ready_wait_seconds_by_request": complete_waits,
        "measured_metrics": measured_metrics,
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
