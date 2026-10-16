#!/usr/bin/env python3
"""Launch M*, run the staggered BAGEL trace workload, and stop all children."""

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
    tail,
)
from experiment_runtime import (
    handle_interrupts, progress, require_free_port, require_local_storage,
    run_logged, stop_owned_group, wait_for_server,
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
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-requests", type=int, default=6)
    parser.add_argument("--stagger-seconds", type=float, default=8.0)
    parser.add_argument("--cfg-branch-policy", choices=("reference", "drop_image"), default="reference")
    parser.add_argument("--base-seed", type=int, default=20261016)
    parser.add_argument("--measured-seeds")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--capture-latents", action="store_true",
                        help="Diagnostic gate only; synchronizes selected steps, invalidates performance timing")
    parser.add_argument(
        "--capture-latent-steps",
        default="0,20,40,41,48",
        help=(
            "Comma-separated post-step indexes to save when --capture-latents "
            "is enabled. Diagnostic only; each dump synchronizes the GPU."
        ),
    )
    parser.add_argument(
        "--image-batch-max-size",
        type=int,
        default=1,
        help=(
            "Experimental true image_gen batch cap; 1 preserves the serial "
            "baseline"
        ),
    )
    parser.add_argument(
        "--image-gen-max-iters",
        type=int,
        help=(
            "Diagnostic early stop for the graph loop. The model retains its "
            "original num_timesteps and timestep schedule."
        ),
    )
    parser.add_argument(
        "--capture-stages",
        action="store_true",
        help="Save selected internal image-refinement tensors; diagnostic only",
    )
    parser.add_argument(
        "--capture-stage-steps",
        default="0,1",
        help="Comma-separated refinement steps for internal stage capture",
    )
    parser.add_argument(
        "--capture-stage-request-prefix",
        default="bagel-measured-00-",
        help=(
            "Only matching measured requests receive full stage fingerprints; "
            "other measured requests receive a compact step-0 input probe"
        ),
    )
    parser.add_argument("--server-start-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=1800.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.image_batch_max_size < 1:
        raise ValueError("--image-batch-max-size must be at least 1")
    if args.image_gen_max_iters is not None and not 1 <= args.image_gen_max_iters <= 49:
        raise ValueError("--image-gen-max-iters must be in 1..49")
    try:
        capture_steps = tuple(
            int(value) for value in args.capture_latent_steps.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError("--capture-latent-steps must be comma-separated integers") from exc
    if args.capture_latents and (
        not capture_steps or len(set(capture_steps)) != len(capture_steps)
        or any(step < 0 or step > 48 for step in capture_steps)
    ):
        raise ValueError("Captured BAGEL latent steps must be unique values in 0..48")
    try:
        stage_capture_steps = tuple(
            int(value) for value in args.capture_stage_steps.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError("--capture-stage-steps must contain integers") from exc
    if args.capture_stages and (
        not stage_capture_steps
        or len(set(stage_capture_steps)) != len(stage_capture_steps)
        or any(step < 0 or step > 48 for step in stage_capture_steps)
    ):
        raise ValueError("Captured BAGEL stage steps must be unique values in 0..48")
    if (
        args.image_gen_max_iters is not None
        and any(step >= args.image_gen_max_iters for step in capture_steps)
        and args.capture_latents
    ):
        raise ValueError("Latent capture steps must be below --image-gen-max-iters")
    if (
        args.image_gen_max_iters is not None
        and any(step >= args.image_gen_max_iters for step in stage_capture_steps)
        and args.capture_stages
    ):
        raise ValueError("Stage capture steps must be below --image-gen-max-iters")
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
    repo_root = args.repo_root.resolve()
    config = args.config if args.config.is_absolute() else repo_root / args.config
    output_dir = args.output_dir.resolve()
    require_local_storage(output_dir, args.cache_dir, repo_root)
    require_free_port(args.port)
    if args.num_requests < 1 or args.warmup_requests < 1 or args.stagger_seconds < 0:
        raise ValueError("Request counts must be positive and stagger must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = output_dir / "execution_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = output_dir / "server.log"
    request_profiles_path = output_dir / "request_profiles.log"
    client_log_path = output_dir / "profile_client.log"
    summary_path = output_dir / "run_summary.json"
    base_url = f"http://127.0.0.1:{args.port}"
    unique = f"{os.getpid()}_{int(time.time())}"
    socket_prefix = f"/tmp/mstar_bagel_profile_{unique}/"
    upload_dir = f"/tmp/mstar_bagel_profile_uploads_{unique}/"
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
        str(repo_root / "experiments/bagel/scripts/profile_staggered.py"),
        "--base-url",
        base_url,
        "--output-dir",
        str(output_dir),
        "--num-requests",
        str(args.num_requests),
        "--stagger-seconds",
        str(args.stagger_seconds),
        "--timeout-seconds",
        str(args.request_timeout_seconds),
        "--cfg-branch-policy", args.cfg_branch_policy,
        "--base-seed", str(args.base_seed),
        "--warmup-requests", str(args.warmup_requests),
    ]
    if args.measured_seeds:
        client_command.extend(["--measured-seeds", args.measured_seeds])
    env = os.environ.copy()
    # Keep the invoked interpreter path instead of resolving its symlink: uv
    # virtual environments may point at a system interpreter, while tools such
    # as ninja are installed beside the venv's Python executable.
    venv_bin = str(Path(sys.executable).parent)
    inherited_path = env.get("PATH")
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MSTAR_EXECUTION_TRACE_DIR": str(trace_dir),
            "MSTAR_BAGEL_IMAGE_BATCH_MAX_SIZE": str(
                args.image_batch_max_size
            ),
            "MSTAR_RUST_ZMQ": "0",
            "PATH": (venv_bin if not inherited_path else os.pathsep.join((venv_bin, inherited_path))),
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "PYTHONPATH": os.pathsep.join(filter(None, (str(repo_root), env.get("PYTHONPATH")))),
        }
    )
    if args.eager:
        env["TORCH_COMPILE_DISABLE"] = "1"
    env.pop("MSTAR_BAGEL_DEBUG_LATENT_DIR", None)
    env.pop("MSTAR_BAGEL_DEBUG_LATENT_STEPS", None)
    env.pop("MSTAR_BAGEL_DEBUG_STAGE_DIR", None)
    env.pop("MSTAR_BAGEL_DEBUG_STAGE_STEPS", None)
    env.pop("MSTAR_BAGEL_DEBUG_STAGE_REQUEST_PREFIX", None)
    env.pop("MSTAR_BAGEL_IMAGE_GEN_MAX_ITERS", None)
    if args.image_gen_max_iters is not None:
        env["MSTAR_BAGEL_IMAGE_GEN_MAX_ITERS"] = str(args.image_gen_max_iters)
    if args.capture_latents:
        env["MSTAR_BAGEL_DEBUG_LATENT_DIR"] = str(output_dir / "debug_latents")
        env["MSTAR_BAGEL_DEBUG_LATENT_STEPS"] = ",".join(
            str(step) for step in capture_steps
        )
    if args.capture_stages:
        env["MSTAR_BAGEL_DEBUG_STAGE_DIR"] = str(output_dir / "debug_stages")
        env["MSTAR_BAGEL_DEBUG_STAGE_STEPS"] = ",".join(
            str(step) for step in stage_capture_steps
        )
        env["MSTAR_BAGEL_DEBUG_STAGE_REQUEST_PREFIX"] = (
            args.capture_stage_request_prefix
        )
    started_utc = datetime.now(timezone.utc)
    summary = {
        "status": "running",
        "stage": "server_startup",
        "started_utc": started_utc.isoformat(),
        "repo": {
            "head": command_result(["git", "rev-parse", "HEAD"], repo_root),
            "branch": command_result(["git", "branch", "--show-current"], repo_root),
            "status": command_result(["git", "status", "--porcelain=v1"], repo_root),
        },
        "config": {"path": str(config), "sha256": file_sha256(config)},
        "workload": {
            "warmup_requests": args.warmup_requests,
            "measured_requests": args.num_requests,
            "stagger_seconds": args.stagger_seconds,
            "image_batch_max_size": args.image_batch_max_size,
            "cfg_branch_policy": args.cfg_branch_policy,
            "base_seed": args.base_seed,
            "measured_seeds": (
                measured_seeds if measured_seeds is not None else None
            ),
            "latent_capture_enabled": args.capture_latents,
            "latent_capture_steps": list(capture_steps) if args.capture_latents else [],
            "stage_capture_enabled": args.capture_stages,
            "stage_capture_steps": (
                list(stage_capture_steps) if args.capture_stages else []
            ),
            "stage_capture_request_prefix": (
                args.capture_stage_request_prefix if args.capture_stages else None
            ),
            "image_gen_max_iters": args.image_gen_max_iters,
            "configured_num_timesteps": 50,
            "torch_compile_disabled": env.get("TORCH_COMPILE_DISABLE") == "1",
        },
        "server_command": server_command,
        "client_command": client_command,
        "environment_overrides": {
            key: env[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "MSTAR_EXECUTION_TRACE_DIR",
                "MSTAR_BAGEL_IMAGE_BATCH_MAX_SIZE",
                "MSTAR_BAGEL_IMAGE_GEN_MAX_ITERS",
                "MSTAR_BAGEL_DEBUG_STAGE_REQUEST_PREFIX",
                "MSTAR_RUST_ZMQ",
                "PATH",
                "PYTHONUNBUFFERED",
            ) if key in env
        },
        "nvidia_smi_before": command_result(["nvidia-smi"], repo_root),
    }
    atomic_json(summary_path, summary)
    process = None
    caught: BaseException | None = None
    server_log = server_log_path.open("ab", buffering=0)
    try:
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
        atomic_json(summary_path, summary)
        summary["server_ready_seconds"] = wait_for_server(
            process, base_url, server_log_path, timeout=args.server_start_timeout_seconds)
        summary["stage"] = "warmup_and_measurement"
        atomic_json(summary_path, summary)
        run_logged(client_command, client_log_path, cwd=repo_root, env=env,
                   stage=f"B={args.image_batch_max_size} {args.cfg_branch_policy}", server=process,
                   timeout=(args.num_requests + args.warmup_requests) * args.request_timeout_seconds
                   + (args.num_requests + args.warmup_requests) * args.stagger_seconds + 120)
        summary["profile_client_returncode"] = 0
        profile_summary = json.loads((output_dir / "profile_summary.json").read_text())
        if profile_summary.get("status") != "success":
            raise RuntimeError(f"profile summary status is {profile_summary.get('status')!r}")
        trace_files = sorted(trace_dir.glob("*.jsonl"))
        if not trace_files or not any(path.stat().st_size for path in trace_files):
            raise RuntimeError("execution trace contains no worker records")
        summary["execution_trace_files"] = [str(path) for path in trace_files]
        summary["status"] = "success"
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        caught = exc
    finally:
        if process is not None:
            try:
                unexpected_exit = process.poll() is not None
                summary["shutdown"] = stop_owned_group(process, args.shutdown_timeout_seconds)
                if summary["status"] == "success" and (
                    unexpected_exit or summary["shutdown"]["method"] != "sigterm"
                ):
                    summary["status"] = "failed_shutdown"
            except Exception as exc:  # noqa: BLE001 - preserve primary error
                summary["shutdown"] = {
                    "method": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if summary["status"] == "success":
                    summary["status"] = "failed_shutdown"
        server_log.close()
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["elapsed_seconds"] = (datetime.fromisoformat(summary["finished_utc"]) - started_utc).total_seconds()
        summary["nvidia_smi_after"] = command_result(["nvidia-smi"], repo_root)
        summary["server_log_tail"] = tail(server_log_path)
        summary["stage"] = "finished"
        atomic_json(summary_path, summary)
        progress(f"Profile {summary['status']}: {output_dir}")
    if caught is not None:
        raise caught
    if summary["status"] != "success":
        return 1
    return 0


if __name__ == "__main__":
    with handle_interrupts():
        raise SystemExit(main())
