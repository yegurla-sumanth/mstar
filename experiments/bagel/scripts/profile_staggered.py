#!/usr/bin/env python3
"""Run one warm-up and a small staggered BAGEL text-to-image workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from smoke_image import GPUSampler, atomic_json, validate_png
from experiment_runtime import progress, require_local_storage


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, event_type: str, **fields) -> None:
        row = {
            "event_type": event_type,
            "perf_counter_ns": time.perf_counter_ns(),
            "wall_time_ns": time.time_ns(),
            **fields,
        }
        with self._lock:
            self._handle.write(json.dumps(row, sort_keys=True) + "\n")
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=6)
    parser.add_argument("--stagger-seconds", type=float, default=8.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--gpu-sample-period-seconds", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        default="A red panda reading a systems paper in a university library",
    )
    parser.add_argument("--base-seed", type=int, default=20261016)
    parser.add_argument(
        "--measured-seeds",
        help="Optional comma-separated seed per measured request",
    )
    parser.add_argument("--cfg-branch-policy", choices=("reference", "drop_image"), default="reference")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    return parser.parse_args()


def run_request(
    *,
    base_url: str,
    output_dir: Path,
    request_id: str,
    request_index: int,
    phase: str,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    timeout_seconds: float,
    target_perf_ns: int,
    events: JsonlWriter,
    cfg_branch_policy: str = "reference",
) -> dict:
    from mstar import MStarClient

    request_dir = output_dir / "requests" / request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    client = MStarClient(base_url, timeout=timeout_seconds)
    remaining = (target_perf_ns - time.perf_counter_ns()) / 1e9
    if remaining > 0:
        time.sleep(remaining)

    submit_perf_ns = time.perf_counter_ns()
    submit_wall_ns = time.time_ns()
    events.write(
        "request_submit",
        request_id=request_id,
        request_index=request_index,
        phase=phase,
        seed=seed,
        target_perf_ns=target_perf_ns,
        submit_perf_ns=submit_perf_ns,
        submit_wall_ns=submit_wall_ns,
    )
    record = {
        "status": "failed",
        "request_id": request_id,
        "request_index": request_index,
        "phase": phase,
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "cfg_branch_policy": cfg_branch_policy,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.4, 1.0],
        "target_perf_ns": target_perf_ns,
        "submit_perf_ns": submit_perf_ns,
        "submit_wall_ns": submit_wall_ns,
        "submission_lag_seconds": max(0, (submit_perf_ns - target_perf_ns) / 1e9),
    }
    try:
        result = client.generate(
            text=prompt,
            output_modalities=("image",),
            request_id=request_id,
            seed=seed,
            width=width,
            height=height,
            think_mode=False,
            cfg_branch_policy=cfg_branch_policy,
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_interval=[0.4, 1.0],
        )
        complete_perf_ns = time.perf_counter_ns()
        complete_wall_ns = time.time_ns()
        if len(result.images) != 1 or not result.images[0]:
            raise RuntimeError(f"expected one nonempty image, received {len(result.images)}")
        image = result.images[0]
        image_path = request_dir / "image.png"
        image_path.write_bytes(image)
        validation = validate_png(image_path, width, height)
        record.update(
            {
                "status": "success",
                "server_request_id": result.request_id,
                "complete_perf_ns": complete_perf_ns,
                "complete_wall_ns": complete_wall_ns,
                "client_elapsed_seconds": (complete_perf_ns - submit_perf_ns) / 1e9,
                "image": {
                    "path": str(image_path),
                    "bytes": len(image),
                    "sha256": hashlib.sha256(image).hexdigest(),
                    "validation": validation,
                },
            }
        )
        events.write(
            "request_complete",
            request_id=request_id,
            request_index=request_index,
            phase=phase,
            complete_perf_ns=complete_perf_ns,
            complete_wall_ns=complete_wall_ns,
            client_elapsed_seconds=record["client_elapsed_seconds"],
        )
    except Exception as exc:
        record.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "complete_perf_ns": time.perf_counter_ns(),
                "complete_wall_ns": time.time_ns(),
            }
        )
        record["client_elapsed_seconds"] = (record["complete_perf_ns"] - submit_perf_ns) / 1e9
        events.write(
            "request_error",
            request_id=request_id,
            request_index=request_index,
            phase=phase,
            error_type=record["error_type"],
            error=record["error"],
        )
    atomic_json(request_dir / "result.json", record)
    progress(f"{phase} request {request_index}: {record['status']} in {record['client_elapsed_seconds']:.1f}s")
    return record


def main() -> int:
    args = parse_args()
    if args.num_requests < 1 or args.warmup_requests < 1:
        raise ValueError("Request and warm-up counts must be positive")
    if args.stagger_seconds < 0:
        raise ValueError("--stagger-seconds must be non-negative")
    measured_seeds = None
    if args.measured_seeds:
        try:
            measured_seeds = [
                int(value) for value in args.measured_seeds.split(",")
                if value.strip()
            ]
        except ValueError as exc:
            raise ValueError("--measured-seeds must contain integers") from exc
        if len(measured_seeds) != args.num_requests:
            raise ValueError(
                "--measured-seeds must contain exactly --num-requests values"
            )

    require_local_storage(args.output_dir)
    from mstar import MStarClient

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = MStarClient(args.base_url, timeout=args.timeout_seconds)
    if not client.health():
        raise RuntimeError(f"server is not healthy at {args.base_url}")

    run_tag = uuid.uuid4().hex[:8]
    events = JsonlWriter(args.output_dir / "client_events.jsonl")
    sampler = GPUSampler(args.gpu_sample_period_seconds)
    summary = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "workload": {
            "warmup_requests": args.warmup_requests,
            "measured_requests": args.num_requests,
            "stagger_seconds": args.stagger_seconds,
            "prompt": args.prompt,
            "base_seed": args.base_seed,
            "measured_seeds": (
                measured_seeds
                if measured_seeds is not None
                else [args.base_seed + index for index in range(args.num_requests)]
            ),
            "width": args.width,
            "height": args.height,
            "expected_num_timesteps": 50,
            "cfg_branch_policy": args.cfg_branch_policy,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": [0.4, 1.0],
        },
    }
    atomic_json(args.output_dir / "profile_summary.json", summary)

    def cohort(phase, count, start):
        rows = []
        summary[f"{phase}_requests"] = rows
        progress(f"Starting {phase}: {count} requests, {args.stagger_seconds}s apart")
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [executor.submit(
                run_request,
                base_url=args.base_url, output_dir=args.output_dir,
                request_id=f"bagel-{phase}-{index:02d}-{run_tag}",
                request_index=index if phase == "measured" else -index - 1,
                phase=phase, prompt=args.prompt,
                seed=(
                    measured_seeds[index]
                    if phase == "measured" and measured_seeds is not None
                    else args.base_seed + index
                    if phase == "measured"
                    else args.base_seed - index - 1
                ),
                width=args.width, height=args.height,
                timeout_seconds=args.timeout_seconds,
                target_perf_ns=start + int(index * args.stagger_seconds * 1e9),
                events=events, cfg_branch_policy=args.cfg_branch_policy,
            ) for index in range(count)]
            for future in as_completed(futures):
                rows.append(future.result())
                rows.sort(key=lambda row: row["request_index"])
                atomic_json(args.output_dir / "profile_summary.json", summary)
        if any(row["status"] != "success" for row in rows):
            raise RuntimeError(f"{phase} cohort contains failed requests; see per-request result.json")
        return rows

    try:
        with sampler:
            warmups = cohort("warmup", args.warmup_requests, time.perf_counter_ns())
            summary["warmup"] = warmups[0]  # retain compatibility with earlier plotting

            measurement_start_perf_ns = time.perf_counter_ns()
            measurement_start_wall_ns = time.time_ns()
            summary["measurement_start_perf_ns"] = measurement_start_perf_ns
            summary["measurement_start_wall_ns"] = measurement_start_wall_ns
            atomic_json(args.output_dir / "profile_summary.json", summary)

            measured = cohort("measured", args.num_requests, measurement_start_perf_ns)

        measured.sort(key=lambda row: row["request_index"])
        summary["measured_requests"] = measured
        summary["measurement_finished_perf_ns"] = max(row["complete_perf_ns"] for row in measured)
        summary["measurement_finished_wall_ns"] = max(row["complete_wall_ns"] for row in measured)
        summary["status"] = "success" if all(row["status"] == "success" for row in measured) else "failed"
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        raise
    finally:
        events.close()
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["gpu_sampling"] = {
            "period_seconds": args.gpu_sample_period_seconds,
            "sample_count": len(sampler.samples),
            "limitation": (
                "External nvidia-smi samples; diagnostic utilization and total "
                "device memory, not allocator or per-request state accounting."
            ),
        }
        atomic_json(args.output_dir / "gpu_samples.json", {"samples": sampler.samples})
        atomic_json(args.output_dir / "profile_summary.json", summary)
    progress(f"Profile client {summary['status']}: {args.output_dir}")
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
