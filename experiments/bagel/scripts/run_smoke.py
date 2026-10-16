#!/usr/bin/env python3
"""Launch pinned single-GPU M*, run one image smoke, and always shut it down."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command_result(command: list[str], cwd: Path) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
            payload = json.loads(response.read())
        return response.status == 200 and payload.get("status") == "healthy"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def tail(path: Path, max_bytes: int = 20000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def stop_process_group(process: subprocess.Popen, timeout_seconds: float) -> dict:
    if process.poll() is not None:
        return {"method": "already_exited", "returncode": process.returncode}
    os.killpg(process.pid, signal.SIGTERM)
    try:
        return {
            "method": "sigterm",
            "returncode": process.wait(timeout=timeout_seconds),
        }
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return {
            "method": "sigkill_after_timeout",
            "returncode": process.wait(timeout=30),
        }


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
    parser.add_argument("--server-start-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--prompt", default="A red panda reading a systems paper in a university library")
    parser.add_argument("--seed", type=int, default=20261016)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config = args.config if args.config.is_absolute() else repo_root / args.config
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = output_dir / "server.log"
    profile_path = output_dir / "request_profiles.log"
    summary_path = output_dir / "run_summary.json"
    base_url = f"http://127.0.0.1:{args.port}"
    unique = f"{os.getpid()}_{int(time.time())}"
    socket_prefix = f"/tmp/mstar_bagel_{unique}/"
    upload_dir = f"/tmp/mstar_bagel_uploads_{unique}/"
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
        str(profile_path),
    ]
    smoke_command = [
        sys.executable,
        str(repo_root / "experiments/bagel/scripts/smoke_image.py"),
        "--base-url",
        base_url,
        "--output-dir",
        str(output_dir),
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--timeout-seconds",
        str(args.request_timeout_seconds),
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MSTAR_RUST_ZMQ": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started_utc = datetime.now(timezone.utc)
    summary = {
        "status": "failed",
        "started_utc": started_utc.isoformat(),
        "repo": {
            "head": command_result(["git", "rev-parse", "HEAD"], repo_root),
            "branch": command_result(["git", "branch", "--show-current"], repo_root),
            "status": command_result(["git", "status", "--porcelain=v1"], repo_root),
        },
        "config": {
            "path": str(config),
            "sha256": file_sha256(config),
        },
        "server_command": server_command,
        "smoke_command": smoke_command,
        "environment_overrides": {
            key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "MSTAR_RUST_ZMQ", "PYTHONUNBUFFERED")
        },
        "nvidia_smi_before": command_result(["nvidia-smi"], repo_root),
    }
    atomic_json(summary_path, summary)
    process = None
    caught: Exception | None = None
    server_log = server_log_path.open("ab", buffering=0)
    try:
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
                    f"server exited before health with code {process.returncode}\n{tail(server_log_path)}"
                )
            if healthy(base_url):
                break
            time.sleep(2)
        else:
            raise TimeoutError(
                f"server did not become healthy within "
                f"{args.server_start_timeout_seconds:.0f}s\n{tail(server_log_path)}"
            )
        summary["server_ready_seconds"] = time.perf_counter() - server_started
        atomic_json(summary_path, summary)
        smoke_log_path = output_dir / "smoke_client.log"
        with smoke_log_path.open("wb") as smoke_log:
            smoke = subprocess.run(
                smoke_command,
                cwd=repo_root,
                env=env,
                stdout=smoke_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        summary["smoke_returncode"] = smoke.returncode
        if smoke.returncode != 0:
            raise RuntimeError(f"smoke client failed with code {smoke.returncode}\n{tail(smoke_log_path)}")
        summary["status"] = "success"
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        caught = exc
    finally:
        if process is not None:
            try:
                summary["shutdown"] = stop_process_group(process, args.shutdown_timeout_seconds)
            except Exception as exc:  # noqa: BLE001 - preserve primary error
                summary["shutdown"] = {
                    "method": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        server_log.close()
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["elapsed_seconds"] = (datetime.fromisoformat(summary["finished_utc"]) - started_utc).total_seconds()
        summary["nvidia_smi_after"] = command_result(["nvidia-smi"], repo_root)
        summary["server_log_tail"] = tail(server_log_path)
        atomic_json(summary_path, summary)
    if caught is not None:
        raise caught
    if summary.get("shutdown", {}).get("method") not in {"sigterm", "already_exited"}:
        summary["status"] = "failed_shutdown"
        atomic_json(summary_path, summary)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
