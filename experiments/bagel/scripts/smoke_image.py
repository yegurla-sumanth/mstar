#!/usr/bin/env python3
"""Submit one fixed BAGEL image request and validate the returned PNG."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gpu_sample() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"sample_time_ns": time.time_ns(), "error": repr(exc)}
    line = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not line:
        return {
            "sample_time_ns": time.time_ns(),
            "returncode": completed.returncode,
            "error": completed.stderr.strip() or "no nvidia-smi row",
        }
    fields = [part.strip() for part in line[0].split(",")]
    if len(fields) != 5:
        return {"sample_time_ns": time.time_ns(), "error": line[0]}
    timestamp, name, used_mib, total_mib, utilization = fields
    return {
        "sample_time_ns": time.time_ns(),
        "nvidia_timestamp": timestamp,
        "name": name,
        "memory_used_mib": int(used_mib),
        "memory_total_mib": int(total_mib),
        "utilization_gpu_percent": int(utilization),
    }


class GPUSampler:
    def __init__(self, period_seconds: float):
        self.period_seconds = period_seconds
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(gpu_sample())
            self._stop.wait(self.period_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.period_seconds * 2))


def validate_png(path: Path, width: int, height: int) -> dict:
    from PIL import Image

    with Image.open(path) as image:
        detected_format = image.format
        detected_size = list(image.size)
        detected_mode = image.mode
        image.verify()
    with Image.open(path) as image:
        image.load()
        extrema = image.getextrema()
    if detected_format != "PNG":
        raise ValueError(f"expected PNG, got {detected_format!r}")
    if detected_size != [width, height]:
        raise ValueError(f"expected dimensions {[width, height]}, got {detected_size}")
    return {
        "format": detected_format,
        "size": detected_size,
        "mode": detected_mode,
        "extrema": extrema,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="A red panda reading a systems paper in a university library")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--gpu-sample-period-seconds", type=float, default=1.0)
    parser.add_argument("--think-mode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from mstar import MStarClient

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "image.png"
    metadata_path = args.output_dir / "smoke_result.json"
    request_id = f"bagel-smoke-{uuid.uuid4()}"
    settings = {
        "request_id": request_id,
        "prompt": args.prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "think_mode": args.think_mode,
        "target_output": "image",
        "expected_num_timesteps": 50,
        "precision": "M* BAGEL default (BF16 weights)",
    }
    atomic_json(args.output_dir / "request.json", settings)
    client = MStarClient(args.base_url, timeout=args.timeout_seconds)
    if not client.health():
        raise RuntimeError(f"server is not healthy at {args.base_url}")

    started_utc = datetime.now(timezone.utc)
    started = time.perf_counter()
    payload = {"status": "failed", "request": settings}
    sampler = GPUSampler(args.gpu_sample_period_seconds)
    try:
        with sampler:
            result = client.generate(
                text=args.prompt,
                output_modalities=("image",),
                request_id=request_id,
                seed=args.seed,
                width=args.width,
                height=args.height,
                think_mode=args.think_mode,
            )
        elapsed = time.perf_counter() - started
        if len(result.images) != 1:
            raise RuntimeError(f"expected one image, received {len(result.images)}")
        image_bytes = result.images[0]
        if not image_bytes:
            raise RuntimeError("server returned an empty image")
        image_path.write_bytes(image_bytes)
        validation = validate_png(image_path, args.width, args.height)
        memory_values = [row["memory_used_mib"] for row in sampler.samples if "memory_used_mib" in row]
        payload = {
            "status": "success",
            "request": settings,
            "server_request_id": result.request_id,
            "started_utc": started_utc.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "client_elapsed_seconds": elapsed,
            "timing_boundary": "client call immediately before POST through complete aggregated response",
            "output": {
                "path": str(image_path),
                "bytes": len(image_bytes),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "validation": validation,
            },
            "gpu_sampling": {
                "period_seconds": args.gpu_sample_period_seconds,
                "sample_count": len(sampler.samples),
                "peak_memory_used_mib": max(memory_values) if memory_values else None,
                "limitation": "external nvidia-smi sampling; diagnostic only, not allocator or live-state accounting",
            },
        }
        atomic_json(metadata_path, payload)
        atomic_json(args.output_dir / "gpu_samples.json", {"samples": sampler.samples})
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "started_utc": started_utc.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "client_elapsed_seconds": time.perf_counter() - started,
            }
        )
        atomic_json(metadata_path, payload)
        atomic_json(args.output_dir / "gpu_samples.json", {"samples": sampler.samples})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
