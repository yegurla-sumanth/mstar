#!/usr/bin/env python3
"""Run paired serial/batched BAGEL profiles and compare their outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_smoke import atomic_json, command_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-requests", type=int, default=2)
    parser.add_argument("--stagger-seconds", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--request-timeout-seconds", type=float, default=1800.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_requests < 2:
        raise ValueError("--num-requests must be at least 2")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    serial_dir = output_dir / "serial_b1"
    batched_dir = output_dir / f"heterogeneous_b{args.batch_size}"
    summary_path = output_dir / "parity_orchestrator.json"
    profile_script = repo_root / "experiments/bagel/scripts/run_profile.py"
    compare_script = repo_root / "experiments/bagel/scripts/compare_batch_parity.py"
    test_scripts = [
        repo_root / "test/modular/test_bagel_image_batch_packing.py",
        repo_root / "test/modular/test_bagel_batch_parity.py",
    ]

    def profile_command(target: Path, batch_size: int) -> list[str]:
        return [
            sys.executable,
            str(profile_script),
            "--repo-root",
            str(repo_root),
            "--cache-dir",
            str(args.cache_dir.resolve()),
            "--output-dir",
            str(target),
            "--port",
            str(args.port),
            "--num-requests",
            str(args.num_requests),
            "--stagger-seconds",
            str(args.stagger_seconds),
            "--image-batch-max-size",
            str(batch_size),
            "--request-timeout-seconds",
            str(args.request_timeout_seconds),
        ]

    unit_commands = [
        [sys.executable, str(test_script), "-v"]
        for test_script in test_scripts
    ]
    serial_command = profile_command(serial_dir, 1)
    batched_command = profile_command(batched_dir, args.batch_size)
    compare_command = [
        sys.executable,
        str(compare_script),
        "--serial-dir",
        str(serial_dir),
        "--batched-dir",
        str(batched_dir),
        "--output",
        str(output_dir / "serial_batch_parity.json"),
    ]
    summary = {
        "status": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "head": command_result(["git", "rev-parse", "HEAD"], repo_root),
            "branch": command_result(
                ["git", "branch", "--show-current"], repo_root
            ),
            "status": command_result(
                ["git", "status", "--porcelain=v1"], repo_root
            ),
        },
        "workload": {
            "num_requests": args.num_requests,
            "stagger_seconds": args.stagger_seconds,
            "serial_batch_size": 1,
            "experimental_batch_size": args.batch_size,
        },
        "commands": {
            "unit": unit_commands,
            "serial": serial_command,
            "batched": batched_command,
            "compare": compare_command,
        },
        "result_dirs": {
            "serial": str(serial_dir),
            "batched": str(batched_dir),
        },
    }
    atomic_json(summary_path, summary)
    started = time.perf_counter()
    env = os.environ.copy()
    prior_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not prior_pythonpath
        else os.pathsep.join((str(repo_root), prior_pythonpath))
    )
    try:
        for unit_command in unit_commands:
            subprocess.run(unit_command, cwd=repo_root, env=env, check=True)
        subprocess.run(serial_command, cwd=repo_root, env=env, check=True)
        subprocess.run(batched_command, cwd=repo_root, env=env, check=True)
        subprocess.run(compare_command, cwd=repo_root, env=env, check=True)
        summary["comparison"] = json.loads(
            (output_dir / "serial_batch_parity.json").read_text()
        )
        summary["status"] = "success"
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        raise
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(summary_path, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
