#!/usr/bin/env python3
"""Run serial, homogeneous-B=2, and heterogeneous-B=2 BAGEL arms."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from experiment_runtime import handle_interrupts, progress, require_local_storage, run_logged
from run_smoke import atomic_json

ALL_STEPS = ",".join(str(step) for step in range(49))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8027)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    require_local_storage(repo, output, args.cache_dir)
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1", TORCH_COMPILE_DISABLE="1")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repo), env.get("PYTHONPATH")))
    )

    summary = {
        "status": "running",
        "stage": "unit_tests",
        "performance_authorized": False,
    }
    started = time.monotonic()
    status_path = output / "diagnostic_orchestrator.json"
    atomic_json(status_path, summary)

    try:
        tests = (
            "test_bagel_image_batch_packing.py",
            "test_execution_trace.py",
            "test_bagel_experiment_runtime.py",
            "test_bagel_pruned_batch_gate.py",
            "test_bagel_batch_shape_diagnostic.py",
        )
        for test in tests:
            run_logged(
                [sys.executable, str(repo / "test/modular" / test), "-v"],
                output / f"{test}.log",
                cwd=repo,
                env=env,
                stage=test,
                timeout=300,
            )

        arms = (
            ("serial_b1", 1, 8.0),
            ("homogeneous_b2", 2, 0.0),
            ("heterogeneous_b2", 2, 8.0),
        )
        for name, cap, stagger in arms:
            summary["stage"] = name
            atomic_json(status_path, summary)
            run_logged(
                [
                    sys.executable,
                    str(repo / "experiments/bagel/scripts/run_profile.py"),
                    "--repo-root", str(repo),
                    "--cache-dir", str(args.cache_dir),
                    "--output-dir", str(output / name),
                    "--port", str(args.port),
                    "--image-batch-max-size", str(cap),
                    "--num-requests", "2",
                    "--warmup-requests", "2",
                    "--stagger-seconds", str(stagger),
                    "--base-seed", "20261016",
                    "--cfg-branch-policy", "drop_image",
                    "--eager",
                    "--capture-latents",
                    "--capture-latent-steps", ALL_STEPS,
                    "--request-timeout-seconds", "600",
                    "--server-start-timeout-seconds", "900",
                ],
                output / f"{name}.log",
                cwd=repo,
                env=env,
                stage=name,
                timeout=3500,
            )

        summary["stage"] = "numerical_analysis"
        atomic_json(status_path, summary)
        report_path = output / "batch_shape_diagnostic.json"
        run_logged(
            [
                sys.executable,
                str(repo / "experiments/bagel/scripts/analyze_batch_shape_diagnostic.py"),
                "--serial-dir", str(output / "serial_b1"),
                "--homogeneous-dir", str(output / "homogeneous_b2"),
                "--heterogeneous-dir", str(output / "heterogeneous_b2"),
                "--output", str(report_path),
            ],
            output / "analysis.log",
            cwd=repo,
            env=env,
            stage="batch-shape numerical analysis",
            timeout=600,
        )
        report = json.loads(report_path.read_text())
        summary.update(
            status="completed",
            diagnostic_status=report["status"],
            hypothesis_assessment=report["hypothesis_assessment"],
        )
    except BaseException as exc:
        summary.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        summary.update(
            stage="finished",
            elapsed_seconds=time.monotonic() - started,
        )
        atomic_json(status_path, summary)
        progress(
            f"Batch-shape diagnostic {summary['status']}: {output}; "
            "no performance sweep was launched"
        )


if __name__ == "__main__":
    with handle_interrupts():
        main()
