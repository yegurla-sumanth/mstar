#!/usr/bin/env python3
"""Launch one M* server and run the paired eager CFG performance pilot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_smoke import (
    atomic_json,
    command_result,
    file_sha256,
    healthy,
    stop_process_group,
    tail,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/bagel/configs/bagel_single_gpu_pinned.yaml"),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8025)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--server-start-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=1800.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    repo_root = args.repo_root.resolve()
    config = args.config if args.config.is_absolute() else repo_root / args.config
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to mix benchmark attempts in existing path: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    trace_dir = output_dir / "execution_trace"
    trace_dir.mkdir()
    server_log_path = output_dir / "server.log"
    request_profiles_path = output_dir / "request_profiles.log"
    client_log_path = output_dir / "cfg_benchmark_client.log"
    analysis_log_path = output_dir / "cfg_benchmark_analysis.log"
    summary_path = output_dir / "cfg_benchmark_orchestrator.json"
    base_url = f"http://127.0.0.1:{args.port}"
    unique = f"{os.getpid()}_{int(time.time())}"
    socket_prefix = f"/tmp/mstar_bagel_cfg_perf_{unique}/"
    upload_dir = f"/tmp/mstar_bagel_cfg_perf_uploads_{unique}/"
    server_command = [
        sys.executable,
        "-m",
        "mstar.api_server.entrypoint",
        "--config",
        str(config),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--cache-dir",
        str(args.cache_dir.resolve()),
        "--socket-path-prefix",
        socket_prefix,
        "--upload-dir",
        upload_dir,
        "--tensor-comm-protocol",
        "SHM",
        "--timeout",
        str(args.request_timeout_seconds),
        "--log-stats-file",
        str(request_profiles_path),
    ]
    client_command = [
        sys.executable,
        str(repo_root / "experiments/bagel/scripts/cfg_branch_benchmark.py"),
        "--base-url",
        base_url,
        "--output-dir",
        str(output_dir),
        "--repeats",
        str(args.repeats),
        "--timeout-seconds",
        str(args.request_timeout_seconds),
    ]
    analysis_command = [
        sys.executable,
        str(repo_root / "experiments/bagel/scripts/analyze_cfg_branch_benchmark.py"),
        "--run-dir",
        str(output_dir),
    ]
    test_commands = [
        [
            sys.executable,
            str(repo_root / "test/modular/test_bagel_image_batch_packing.py"),
            "-v",
        ],
        [
            sys.executable,
            str(repo_root / "test/modular/test_execution_trace.py"),
            "-v",
        ],
    ]

    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    inherited_path = env.get("PATH")
    prior_pythonpath = env.get("PYTHONPATH")
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "MSTAR_EXECUTION_TRACE_DIR": str(trace_dir),
        "MSTAR_BAGEL_IMAGE_BATCH_MAX_SIZE": "1",
        "MSTAR_RUST_ZMQ": "0",
        "MPLBACKEND": "Agg",
        "TORCH_COMPILE_DISABLE": "1",
        "PATH": (
            venv_bin
            if not inherited_path
            else os.pathsep.join((venv_bin, inherited_path))
        ),
        "PYTHONPATH": (
            str(repo_root)
            if not prior_pythonpath
            else os.pathsep.join((str(repo_root), prior_pythonpath))
        ),
        "PYTHONUNBUFFERED": "1",
    })
    env.pop("MSTAR_BAGEL_DEBUG_LATENT_DIR", None)
    env.pop("MSTAR_BAGEL_DEBUG_LATENT_STEPS", None)
    started_utc = datetime.now(timezone.utc)
    summary = {
        "status": "failed",
        "started_utc": started_utc.isoformat(),
        "repo": {
            "head": command_result(["git", "rev-parse", "HEAD"], repo_root),
            "branch": command_result(
                ["git", "branch", "--show-current"], repo_root
            ),
            "status": command_result(
                ["git", "status", "--porcelain=v1"], repo_root
            ),
        },
        "config": {"path": str(config), "sha256": file_sha256(config)},
        "workload": {
            "policies": ["reference", "drop_image"],
            "repeats": args.repeats,
            "warmups_per_policy": 1,
            "image_batch_max_size": 1,
            "torch_compile_disabled": True,
            "latent_capture_enabled": False,
            "timing_status": "paired_eager_performance_pilot",
        },
        "commands": {
            "unit": test_commands,
            "server": server_command,
            "client": client_command,
            "analysis": analysis_command,
        },
        "environment_overrides": {
            key: env[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "MSTAR_EXECUTION_TRACE_DIR",
                "MSTAR_BAGEL_IMAGE_BATCH_MAX_SIZE",
                "MSTAR_RUST_ZMQ",
                "MPLBACKEND",
                "TORCH_COMPILE_DISABLE",
            )
        },
        "nvidia_smi_before": command_result(["nvidia-smi"], repo_root),
    }
    atomic_json(summary_path, summary)
    process = None
    caught: Exception | None = None
    server_log = server_log_path.open("ab", buffering=0)
    try:
        for command in test_commands:
            subprocess.run(command, cwd=repo_root, env=env, check=True)
        server_started = time.perf_counter()
        process = subprocess.Popen(
            server_command,
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        summary["server_pid"] = process.pid
        deadline = time.monotonic() + args.server_start_timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    "server exited before health with code "
                    f"{process.returncode}\n{tail(server_log_path)}"
                )
            if healthy(base_url):
                break
            time.sleep(2)
        else:
            raise TimeoutError(
                "server did not become healthy within "
                f"{args.server_start_timeout_seconds:.0f}s\n"
                f"{tail(server_log_path)}"
            )
        summary["server_ready_seconds"] = time.perf_counter() - server_started
        atomic_json(summary_path, summary)

        with client_log_path.open("wb") as client_log:
            client = subprocess.run(
                client_command,
                cwd=repo_root,
                env=env,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        summary["client_returncode"] = client.returncode
        if client.returncode != 0:
            raise RuntimeError(
                f"CFG benchmark client failed with code {client.returncode}\n"
                f"{tail(client_log_path)}"
            )

        with analysis_log_path.open("wb") as analysis_log:
            analysis = subprocess.run(
                analysis_command,
                cwd=repo_root,
                env=env,
                stdout=analysis_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        summary["analysis_returncode"] = analysis.returncode
        if analysis.returncode != 0:
            raise RuntimeError(
                f"CFG benchmark analysis failed with code {analysis.returncode}\n"
                f"{tail(analysis_log_path)}"
            )
        summary["analysis"] = json.loads(
            (output_dir / "cfg_branch_benchmark_analysis.json").read_text()
        )
        summary["status"] = "success"
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        caught = exc
    finally:
        if process is not None:
            try:
                summary["shutdown"] = stop_process_group(
                    process, args.shutdown_timeout_seconds
                )
            except Exception as exc:  # noqa: BLE001 - preserve primary error
                summary["shutdown"] = {
                    "method": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        server_log.close()
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["elapsed_seconds"] = (
            datetime.fromisoformat(summary["finished_utc"]) - started_utc
        ).total_seconds()
        summary["nvidia_smi_after"] = command_result(
            ["nvidia-smi"], repo_root
        )
        summary["server_log_tail"] = tail(server_log_path)
        atomic_json(summary_path, summary)
    if caught is not None:
        raise caught
    if summary.get("shutdown", {}).get("method") not in {
        "sigterm", "already_exited"
    }:
        summary["status"] = "failed_shutdown"
        atomic_json(summary_path, summary)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
