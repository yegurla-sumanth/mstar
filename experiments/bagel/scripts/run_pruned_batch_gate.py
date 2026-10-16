#!/usr/bin/env python3
"""Two-arm eager gate: identical pruned work at B=1 and heterogeneous B=2."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from experiment_runtime import handle_interrupts, progress, require_local_storage, run_logged
from run_smoke import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8026)
    args = parser.parse_args()
    repo, output = args.repo_root.resolve(), args.output_dir.resolve()
    require_local_storage(repo, output, args.cache_dir)
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1", TORCH_COMPILE_DISABLE="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo), env.get("PYTHONPATH"))))
    summary = {"status": "running", "stage": "unit_tests", "performance_authorized": False}
    started = time.monotonic()
    status_path = output / "gate_orchestrator.json"
    atomic_json(status_path, summary)
    try:
        for test in ("test_bagel_image_batch_packing.py", "test_execution_trace.py",
                     "test_bagel_experiment_runtime.py", "test_bagel_pruned_batch_gate.py"):
            run_logged([sys.executable, str(repo / "test/modular" / test), "-v"],
                       output / f"{test}.log", cwd=repo, env=env, stage=test, timeout=300)
        for cap, name in ((1, "serial_b1"), (2, "heterogeneous_b2")):
            summary["stage"] = name
            atomic_json(status_path, summary)
            run_logged([
                sys.executable, str(repo / "experiments/bagel/scripts/run_profile.py"),
                "--repo-root", str(repo), "--cache-dir", str(args.cache_dir),
                "--output-dir", str(output / name), "--port", str(args.port),
                "--image-batch-max-size", str(cap), "--num-requests", "2",
                "--warmup-requests", "2", "--stagger-seconds", "8",
                "--base-seed", "20261016", "--cfg-branch-policy", "drop_image",
                "--eager", "--capture-latents", "--request-timeout-seconds", "600",
                "--server-start-timeout-seconds", "900",
            ], output / f"{name}.log", cwd=repo, env=env, stage=name, timeout=3500)
        summary["stage"] = "numerical_analysis"
        atomic_json(status_path, summary)
        run_logged([
            sys.executable, str(repo / "experiments/bagel/scripts/analyze_pruned_batch_gate.py"),
            "--serial-dir", str(output / "serial_b1"), "--batched-dir", str(output / "heterogeneous_b2"),
            "--output", str(output / "pruned_batch_gate.json"),
        ], output / "analysis.log", cwd=repo, env=env, stage="numerical analysis", timeout=300)
        summary["status"] = "completed"
        summary["numerical_status"] = json.loads((output / "pruned_batch_gate.json").read_text())["status"]
    except BaseException as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        summary.update(stage="finished", elapsed_seconds=time.monotonic() - started)
        atomic_json(status_path, summary)
        progress(f"Gate {summary['status']}: {output}; no performance sweep was launched")


if __name__ == "__main__":
    with handle_interrupts():
        main()
