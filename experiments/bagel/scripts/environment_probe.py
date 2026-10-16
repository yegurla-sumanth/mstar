#!/usr/bin/env python3
"""Record the host/GPU environment and fail before an unsuitable large download."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "torchcodec",
    "flashinfer-python",
    "huggingface-hub",
    "mstar-ai",
    "ninja",
    "Pillow",
    "transformers",
    "uv",
)


def run(command: list[str]) -> dict:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": repr(exc)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def host_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_details() -> dict:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - the failure is part of the probe
        return {"available": False, "error": repr(exc)}
    details = {
        "available": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        details.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    return details


def gpu_rows() -> tuple[list[dict], dict]:
    query = run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    if query.get("returncode") == 0:
        for line in query["stdout"].splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 5:
                continue
            name, uuid, total_mib, free_mib, driver = fields
            rows.append(
                {
                    "name": name,
                    "uuid": uuid,
                    "memory_total_mib": int(total_mib),
                    "memory_free_mib": int(free_mib),
                    "driver_version": driver,
                }
            )
    return rows, query


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disk-path", type=Path, default=Path.cwd())
    parser.add_argument("--require-gpu-memory-gib", type=float, default=0.0)
    parser.add_argument("--require-host-memory-gib", type=float, default=0.0)
    parser.add_argument("--require-disk-free-gib", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    disk = shutil.disk_usage(args.disk_path)
    host_bytes = host_memory_bytes()
    gpus, nvidia_query = gpu_rows()
    failures = []
    gib = 1024**3
    if not gpus:
        failures.append("nvidia-smi reported no GPU")
    elif gpus[0]["memory_total_mib"] / 1024 < args.require_gpu_memory_gib:
        failures.append(
            f"GPU memory {gpus[0]['memory_total_mib'] / 1024:.1f} GiB is below "
            f"required {args.require_gpu_memory_gib:.1f} GiB"
        )
    if host_bytes is None:
        failures.append("host memory could not be measured")
    elif host_bytes / gib < args.require_host_memory_gib:
        failures.append(
            f"host memory {host_bytes / gib:.1f} GiB is below required {args.require_host_memory_gib:.1f} GiB"
        )
    if disk.free / gib < args.require_disk_free_gib:
        failures.append(f"free disk {disk.free / gib:.1f} GiB is below required {args.require_disk_free_gib:.1f} GiB")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "host_memory_bytes": host_bytes,
        "disk": {
            "path": str(args.disk_path.resolve()),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "gpus": gpus,
        "nvidia_smi_query": nvidia_query,
        "nvidia_smi_full": run(["nvidia-smi"]),
        "tools": {name: shutil.which(name) for name in ("git", "ninja", "nsys", "nvcc", "uv")},
        "packages": package_versions(),
        "torch": torch_details(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "COLAB_GPU",
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "MSTAR_RUST_ZMQ",
                "NVIDIA_VISIBLE_DEVICES",
            )
        },
        "requirements": {
            "gpu_memory_gib": args.require_gpu_memory_gib,
            "host_memory_gib": args.require_host_memory_gib,
            "disk_free_gib": args.require_disk_free_gib,
        },
        "failures": failures,
    }
    if args.output:
        atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
