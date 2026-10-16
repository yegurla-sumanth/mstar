#!/usr/bin/env python3
"""Benchmark exact BAGEL CFG image-branch pruning in paired B=1 runs."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cfg_branch_sweep import run_request
from smoke_image import GPUSampler, atomic_json


POLICIES = ("reference", "drop_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8025")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--gpu-sample-period-seconds", type=float, default=0.5)
    parser.add_argument(
        "--prompt",
        default="A red panda reading a systems paper in a university library",
    )
    parser.add_argument("--base-seed", type=int, default=20261016)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 for dispersion reporting")

    from mstar import MStarClient

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = MStarClient(args.base_url, timeout=args.timeout_seconds)
    if not client.health():
        raise RuntimeError(f"server is not healthy at {args.base_url}")

    run_tag = uuid.uuid4().hex[:8]
    summary = {
        "status": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "measurement_boundary": (
            "One loaded server; both policy shapes warmed once; sequential B=1 "
            "requests in alternating paired order; client timing spans generate() "
            "through the complete aggregated response; no latent capture."
        ),
        "workload": {
            "policies": list(POLICIES),
            "repeats": args.repeats,
            "warmup_requests": len(POLICIES),
            "order": "reference/drop_image on even pairs; reversed on odd pairs",
            "prompt": args.prompt,
            "base_seed": args.base_seed,
            "width": args.width,
            "height": args.height,
            "num_timesteps": 50,
            "timestep_shift": 3.0,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": [0.4, 1.0],
        },
        "warmup_requests": [],
        "measured_requests": [],
    }
    summary_path = args.output_dir / "cfg_benchmark_summary.json"
    atomic_json(summary_path, summary)
    sampler = GPUSampler(args.gpu_sample_period_seconds)
    try:
        with sampler:
            for warmup_index, policy in enumerate(POLICIES):
                record = run_request(
                    client=client,
                    output_dir=args.output_dir,
                    request_id=f"bagel-perf-warmup-{policy}-{run_tag}",
                    phase="warmup",
                    repeat=-1,
                    policy=policy,
                    prompt=args.prompt,
                    seed=args.base_seed - 2 + warmup_index,
                    width=args.width,
                    height=args.height,
                )
                summary["warmup_requests"].append(record)
                atomic_json(summary_path, summary)
                if record["status"] != "success":
                    raise RuntimeError(
                        f"{policy} warm-up failed: {record.get('error')}"
                    )

            for repeat in range(args.repeats):
                order = POLICIES if repeat % 2 == 0 else tuple(reversed(POLICIES))
                for policy in order:
                    record = run_request(
                        client=client,
                        output_dir=args.output_dir,
                        request_id=f"bagel-perf-{policy}-r{repeat:02d}-{run_tag}",
                        phase="measured",
                        repeat=repeat,
                        policy=policy,
                        prompt=args.prompt,
                        seed=args.base_seed + repeat,
                        width=args.width,
                        height=args.height,
                    )
                    summary["measured_requests"].append(record)
                    atomic_json(summary_path, summary)
                    if record["status"] != "success":
                        raise RuntimeError(
                            f"{policy} repeat {repeat} failed: {record.get('error')}"
                        )
        summary["status"] = "success"
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        raise
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["gpu_sampling"] = {
            "period_seconds": args.gpu_sample_period_seconds,
            "sample_count": len(sampler.samples),
        }
        atomic_json(summary_path, summary)
        atomic_json(
            args.output_dir / "gpu_samples.json", {"samples": sampler.samples}
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
