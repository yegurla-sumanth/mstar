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
) -> dict:
    from mstar import MStarClient

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
        "target_perf_ns": target_perf_ns,
        "submit_perf_ns": submit_perf_ns,
        "submit_wall_ns": submit_wall_ns,
    }
    request_dir = output_dir / "requests" / request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    try:
        client = MStarClient(base_url, timeout=timeout_seconds)
        result = client.generate(
            text=prompt,
            output_modalities=("image",),
            request_id=request_id,
            seed=seed,
            width=width,
            height=height,
            think_mode=False,
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
    return record


def main() -> int:
    args = parse_args()
    if args.num_requests < 1:
        raise ValueError("--num-requests must be positive")
    if args.stagger_seconds < 0:
        raise ValueError("--stagger-seconds must be non-negative")

    from mstar import MStarClient

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = MStarClient(args.base_url, timeout=args.timeout_seconds)
    if not client.health():
        raise RuntimeError(f"server is not healthy at {args.base_url}")

    run_tag = uuid.uuid4().hex[:8]
    events = JsonlWriter(args.output_dir / "client_events.jsonl")
    sampler = GPUSampler(args.gpu_sample_period_seconds)
    summary = {
        "status": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "workload": {
            "warmup_requests": 1,
            "measured_requests": args.num_requests,
            "stagger_seconds": args.stagger_seconds,
            "prompt": args.prompt,
            "base_seed": args.base_seed,
            "width": args.width,
            "height": args.height,
            "expected_num_timesteps": 50,
        },
    }
    atomic_json(args.output_dir / "profile_summary.json", summary)
    try:
        with sampler:
            warmup_id = f"bagel-warmup-{run_tag}"
            warmup_start = time.perf_counter_ns()
            warmup = run_request(
                base_url=args.base_url,
                output_dir=args.output_dir,
                request_id=warmup_id,
                request_index=-1,
                phase="warmup",
                prompt=args.prompt,
                seed=args.base_seed - 1,
                width=args.width,
                height=args.height,
                timeout_seconds=args.timeout_seconds,
                target_perf_ns=warmup_start,
                events=events,
            )
            summary["warmup"] = warmup
            atomic_json(args.output_dir / "profile_summary.json", summary)
            if warmup["status"] != "success":
                raise RuntimeError(f"warm-up request failed: {warmup.get('error')}")

            measurement_start_perf_ns = time.perf_counter_ns()
            measurement_start_wall_ns = time.time_ns()
            summary["measurement_start_perf_ns"] = measurement_start_perf_ns
            summary["measurement_start_wall_ns"] = measurement_start_wall_ns
            atomic_json(args.output_dir / "profile_summary.json", summary)

            futures = []
            with ThreadPoolExecutor(max_workers=args.num_requests) as executor:
                for index in range(args.num_requests):
                    request_id = f"bagel-measured-{index:02d}-{run_tag}"
                    target_ns = measurement_start_perf_ns + int(index * args.stagger_seconds * 1e9)
                    futures.append(
                        executor.submit(
                            run_request,
                            base_url=args.base_url,
                            output_dir=args.output_dir,
                            request_id=request_id,
                            request_index=index,
                            phase="measured",
                            prompt=args.prompt,
                            seed=args.base_seed + index,
                            width=args.width,
                            height=args.height,
                            timeout_seconds=args.timeout_seconds,
                            target_perf_ns=target_ns,
                            events=events,
                        )
                    )
                measured = [future.result() for future in as_completed(futures)]

        measured.sort(key=lambda row: row["request_index"])
        summary["measured_requests"] = measured
        summary["measurement_finished_perf_ns"] = time.perf_counter_ns()
        summary["measurement_finished_wall_ns"] = time.time_ns()
        summary["status"] = "success" if all(row["status"] == "success" for row in measured) else "failed"
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
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
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
