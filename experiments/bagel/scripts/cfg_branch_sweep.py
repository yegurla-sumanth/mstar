#!/usr/bin/env python3
"""Run sequential BAGEL CFG branch-policy requests on one loaded server."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mstar.model.bagel.batching import CFG_BRANCH_POLICIES

from smoke_image import GPUSampler, atomic_json, validate_png


DEFAULT_POLICIES = [
    "reference",
    "drop_image",
    "late_main",
    "auto",
    "main_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--repeats", type=int, default=1)
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


def run_request(
    *,
    client,
    output_dir: Path,
    request_id: str,
    phase: str,
    repeat: int,
    policy: str,
    prompt: str,
    seed: int,
    width: int,
    height: int,
) -> dict:
    record = {
        "status": "failed",
        "request_id": request_id,
        "phase": phase,
        "repeat": repeat,
        "request_index": repeat,
        "cfg_branch_policy": policy,
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "submit_perf_ns": time.perf_counter_ns(),
        "submit_wall_ns": time.time_ns(),
    }
    request_dir = output_dir / "requests" / request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = client.generate(
            text=prompt,
            output_modalities=("image",),
            request_id=request_id,
            seed=seed,
            width=width,
            height=height,
            think_mode=False,
            cfg_branch_policy=policy,
        )
        complete_perf_ns = time.perf_counter_ns()
        if len(result.images) != 1 or not result.images[0]:
            raise RuntimeError(
                f"expected one nonempty image, received {len(result.images)}"
            )
        image = result.images[0]
        image_path = request_dir / "image.png"
        image_path.write_bytes(image)
        record.update({
            "status": "success",
            "server_request_id": result.request_id,
            "complete_perf_ns": complete_perf_ns,
            "complete_wall_ns": time.time_ns(),
            "client_elapsed_seconds": (
                complete_perf_ns - record["submit_perf_ns"]
            ) / 1e9,
            "image": {
                "path": str(image_path),
                "bytes": len(image),
                "sha256": hashlib.sha256(image).hexdigest(),
                "validation": validate_png(image_path, width, height),
            },
        })
    except Exception as exc:
        record.update({
            "error_type": type(exc).__name__,
            "error": str(exc),
            "complete_perf_ns": time.perf_counter_ns(),
            "complete_wall_ns": time.time_ns(),
        })
        record["client_elapsed_seconds"] = (
            record["complete_perf_ns"] - record["submit_perf_ns"]
        ) / 1e9
    atomic_json(request_dir / "result.json", record)
    return record


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    unknown = set(args.policies).difference(CFG_BRANCH_POLICIES)
    if unknown:
        raise ValueError(f"unsupported CFG branch policies: {sorted(unknown)}")
    if len(set(args.policies)) != len(args.policies):
        raise ValueError("--policies must not contain duplicates")
    if "reference" not in args.policies:
        raise ValueError("--policies must include the reference arm")

    from mstar import MStarClient

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = MStarClient(args.base_url, timeout=args.timeout_seconds)
    if not client.health():
        raise RuntimeError(f"server is not healthy at {args.base_url}")

    run_tag = uuid.uuid4().hex[:8]
    sampler = GPUSampler(args.gpu_sample_period_seconds)
    summary = {
        "status": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "workload": {
            "policies": args.policies,
            "repeats": args.repeats,
            "warmup_requests": 1,
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
        "measured_requests": [],
    }
    atomic_json(args.output_dir / "cfg_sweep_summary.json", summary)
    try:
        with sampler:
            warmup = run_request(
                client=client,
                output_dir=args.output_dir,
                request_id=f"bagel-cfg-warmup-{run_tag}",
                phase="warmup",
                repeat=-1,
                policy="reference",
                prompt=args.prompt,
                seed=args.base_seed - 1,
                width=args.width,
                height=args.height,
            )
            summary["warmup"] = warmup
            atomic_json(args.output_dir / "cfg_sweep_summary.json", summary)
            if warmup["status"] != "success":
                raise RuntimeError(
                    f"warm-up request failed: {warmup.get('error')}"
                )

            for repeat in range(args.repeats):
                order = (
                    args.policies
                    if repeat % 2 == 0
                    else list(reversed(args.policies))
                )
                for policy in order:
                    record = run_request(
                        client=client,
                        output_dir=args.output_dir,
                        request_id=(
                            f"bagel-cfg-{policy}-r{repeat:02d}-{run_tag}"
                        ),
                        phase="measured",
                        repeat=repeat,
                        policy=policy,
                        prompt=args.prompt,
                        seed=args.base_seed + repeat,
                        width=args.width,
                        height=args.height,
                    )
                    summary["measured_requests"].append(record)
                    atomic_json(
                        args.output_dir / "cfg_sweep_summary.json", summary
                    )
                    if record["status"] != "success":
                        raise RuntimeError(
                            f"{policy} repeat {repeat} failed: "
                            f"{record.get('error')}"
                        )
        summary["status"] = "success"
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["gpu_sampling"] = {
            "period_seconds": args.gpu_sample_period_seconds,
            "sample_count": len(sampler.samples),
        }
        atomic_json(
            args.output_dir / "gpu_samples.json", {"samples": sampler.samples}
        )
        atomic_json(args.output_dir / "cfg_sweep_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
