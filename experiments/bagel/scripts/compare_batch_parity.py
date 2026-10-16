#!/usr/bin/env python3
"""Compare identical serial and heterogeneous-batched BAGEL profile outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from smoke_image import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-dir", type=Path, required=True)
    parser.add_argument("--batched-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_summary(root: Path) -> dict:
    path = root / "profile_summary.json"
    summary = json.loads(path.read_text())
    if summary.get("status") != "success":
        raise ValueError(f"profile is not successful: {path}")
    return summary


def measured_by_index(summary: dict) -> dict[int, dict]:
    rows = {}
    for row in summary.get("measured_requests", []):
        if row.get("status") != "success":
            raise ValueError(
                f"request {row.get('request_index')} is not successful"
            )
        index = int(row["request_index"])
        if index in rows:
            raise ValueError(f"duplicate measured request index {index}")
        rows[index] = row
    return rows


def image_array(row: dict) -> np.ndarray:
    with Image.open(row["image"]["path"]) as image:
        return np.asarray(image.convert("RGB"), dtype=np.int16)


def compare_images(serial: dict, batched: dict) -> dict:
    for key in ("prompt", "seed", "width", "height"):
        if serial[key] != batched[key]:
            raise ValueError(
                f"request {serial['request_index']} differs on {key}: "
                f"{serial[key]!r} vs {batched[key]!r}"
            )
    reference = image_array(serial)
    candidate = image_array(batched)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"request {serial['request_index']} image shapes differ: "
            f"{reference.shape} vs {candidate.shape}"
        )
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(np.square(delta)))
    psnr_db = math.inf if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)
    return {
        "request_index": serial["request_index"],
        "seed": serial["seed"],
        "serial_sha256": serial["image"]["sha256"],
        "batched_sha256": batched["image"]["sha256"],
        "sha256_equal": serial["image"]["sha256"] == batched["image"]["sha256"],
        "max_abs_pixel_error": float(np.max(np.abs(delta))),
        "mean_abs_pixel_error": float(np.mean(np.abs(delta))),
        "mse": mse,
        "psnr_db": psnr_db,
        "equal_pixel_fraction": float(np.mean(delta == 0)),
    }


def trace_evidence(root: Path) -> dict:
    rows = []
    for path in sorted((root / "execution_trace").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    image_batches = [
        row for row in rows
        if row.get("event_type") == "component_complete"
        and row.get("node_name") == "LLM"
        and row.get("graph_walk") == "image_gen"
    ]
    multi_request = [row for row in image_batches if row.get("batch_size", 0) > 1]
    heterogeneous = []
    for row in multi_request:
        progress = row.get("dynamic_loop_iter_counts", {})
        signatures = {
            tuple(sorted(counts.items()))
            for counts in progress.values()
        }
        if len(signatures) > 1:
            heterogeneous.append(row)

    return {
        "image_component_completions": len(image_batches),
        "multi_request_image_completions": len(multi_request),
        "heterogeneous_step_image_completions": len(heterogeneous),
        "maximum_image_batch_size": max(
            (int(row.get("batch_size", 0)) for row in image_batches),
            default=0,
        ),
        "heterogeneous_examples": [
            {
                "batch_id": row.get("batch_id"),
                "request_ids": row.get("request_ids"),
                "dynamic_loop_iter_counts": row.get(
                    "dynamic_loop_iter_counts"
                ),
                "cuda_duration_ms": row.get("cuda_duration_ms"),
            }
            for row in heterogeneous[:5]
        ],
    }


def main() -> int:
    args = parse_args()
    serial_dir = args.serial_dir.resolve()
    batched_dir = args.batched_dir.resolve()
    serial_summary = load_summary(serial_dir)
    batched_summary = load_summary(batched_dir)
    serial_rows = measured_by_index(serial_summary)
    batched_rows = measured_by_index(batched_summary)
    if serial_rows.keys() != batched_rows.keys():
        raise ValueError(
            "serial and batched profiles contain different request indexes"
        )

    comparisons = [
        compare_images(serial_rows[index], batched_rows[index])
        for index in sorted(serial_rows)
    ]
    trace = trace_evidence(batched_dir)
    payload = {
        "status": (
            "complete"
            if trace["heterogeneous_step_image_completions"] > 0
            else "missing_heterogeneous_batch_evidence"
        ),
        "comparison_boundary": (
            "same prompt/seed/geometry per request; serial model batches "
            "versus image_gen max batch size greater than one"
        ),
        "serial_dir": str(serial_dir),
        "batched_dir": str(batched_dir),
        "requests": comparisons,
        "summary": {
            "request_count": len(comparisons),
            "exact_image_count": sum(
                int(row["sha256_equal"]) for row in comparisons
            ),
            "minimum_psnr_db": min(
                (row["psnr_db"] for row in comparisons), default=math.inf
            ),
            "maximum_abs_pixel_error": max(
                (row["max_abs_pixel_error"] for row in comparisons), default=0
            ),
        },
        "batched_trace": trace,
        "interpretation": (
            "No numerical acceptance threshold is imposed before the first "
            "measurement. Inspect one-step/trajectory evidence if image "
            "differences exceed ordinary batch-shape drift."
        ),
    }
    output = args.output or batched_dir / "serial_batch_parity.json"
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
