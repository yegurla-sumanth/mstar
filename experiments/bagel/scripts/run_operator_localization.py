#!/usr/bin/env python3
"""Run the bounded BAGEL B=1/B=2 operator-localization diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from experiment_runtime import (
    handle_interrupts,
    progress,
    require_local_storage,
    run_logged,
)
from run_smoke import atomic_json

MAX_ITERS = 8
LATENT_STEPS = ",".join(str(step) for step in range(MAX_ITERS))
STAGE_STEPS = "0,1,2,3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8031)
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
        "configured_num_timesteps": 50,
        "executed_refinement_steps": MAX_ITERS,
        "stage_capture_steps": [0, 1, 2, 3],
    }
    started = time.monotonic()
    status_path = output / "operator_localization_orchestrator.json"
    atomic_json(status_path, summary)

    try:
        tests = (
            "test_bagel_image_batch_packing.py",
            "test_execution_trace.py",
            "test_bagel_experiment_runtime.py",
            "test_bagel_pruned_batch_gate.py",
            "test_bagel_operator_localization.py",
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
            # name, cap, measured seeds, measured count, warm-up count
            ("serial_b1", 1, "20261016", 1, 1),
            ("batched_partner_a", 2, "20261016,20261017", 2, 2),
            ("batched_partner_a_repeat", 2, "20261016,20261017", 2, 2),
            ("batched_partner_b", 2, "20261016,20261117", 2, 2),
        )
        for name, cap, seeds, measured_count, warmup_count in arms:
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
                    "--num-requests", str(measured_count),
                    "--warmup-requests", str(warmup_count),
                    "--stagger-seconds", "0",
                    "--base-seed", "20261016",
                    "--measured-seeds", seeds,
                    "--cfg-branch-policy", "drop_image",
                    "--eager",
                    "--image-gen-max-iters", str(MAX_ITERS),
                    "--capture-latents",
                    "--capture-latent-steps", LATENT_STEPS,
                    "--capture-stages",
                    "--capture-stage-steps", STAGE_STEPS,
                    "--capture-stage-request-prefix", "bagel-measured-00-",
                    "--request-timeout-seconds", "600",
                    "--server-start-timeout-seconds", "900",
                ],
                output / f"{name}.log",
                cwd=repo,
                env=env,
                stage=name,
                timeout=2800,
            )

        summary["stage"] = "operator_analysis"
        atomic_json(status_path, summary)
        report_path = output / "operator_localization.json"
        run_logged(
            [
                sys.executable,
                str(repo / "experiments/bagel/scripts/analyze_operator_localization.py"),
                "--serial-dir", str(output / "serial_b1"),
                "--partner-a-dir", str(output / "batched_partner_a"),
                "--partner-a-repeat-dir", str(output / "batched_partner_a_repeat"),
                "--partner-b-dir", str(output / "batched_partner_b"),
                "--output", str(report_path),
            ],
            output / "analysis.log",
            cwd=repo,
            env=env,
            stage="operator-localization numerical analysis",
            timeout=600,
        )
        report = json.loads(report_path.read_text())
        summary.update(
            status="completed",
            diagnostic_status=report["status"],
            hypothesis_assessment=report["hypothesis_assessment"],
            report_path=str(report_path),
        )
    except BaseException as exc:
        summary.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        summary.update(stage="finished", elapsed_seconds=time.monotonic() - started)
        atomic_json(status_path, summary)
        progress(
            f"Operator-localization diagnostic {summary['status']}: {output}; "
            "no performance sweep was launched"
        )


if __name__ == "__main__":
    with handle_interrupts():
        main()
