"""CPU-only lifecycle helpers for local-disk BAGEL experiments.

Never put mounted Drive I/O on a measured execution path. Back up an archive
only after the server and client have stopped. These helpers do not launch a
model unless explicitly called by an experiment entrypoint.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import queue
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from run_smoke import healthy, tail


def progress(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"PROGRESS [{stamp}] {message}", flush=True)


def require_local_storage(*paths: Path, mountinfo: Path = Path("/proc/self/mountinfo")) -> None:
    """Reject Drive, symlinks into Drive, and known network/FUSE mounts.

    On non-Linux hosts, the explicit Drive-path check still runs. This is a
    conservative working-directory guard, not a guarantee about disk speed.
    """
    mounts = []
    if mountinfo.exists():
        for line in mountinfo.read_text().splitlines():
            before, after = line.split(" - ", 1)
            mount = before.split()[4].replace("\\040", " ").replace("\\134", "\\")
            mounts.append((Path(mount), after.split()[0]))
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_relative_to(Path("/content/drive")):
            raise ValueError(f"Active experiment path must be local, not Drive: {path}")
        matches = [(root, fs) for root, fs in mounts if resolved.is_relative_to(root)]
        if matches:
            _, filesystem = max(matches, key=lambda row: len(row[0].parts))
            if filesystem.startswith(("fuse", "nfs", "cifs", "smb", "9p")):
                raise ValueError(f"Active experiment path uses {filesystem}, not local storage: {path}")


def require_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {port} is busy. No new server was started; do not attach to an old run."
            ) from exc


def stop_owned_group(process: subprocess.Popen, timeout: float = 30) -> dict:
    """Stop only a child launched by us with start_new_session=True.

    A dead parent may leave worker children in its process group. Unlike a
    poll-only check, killpg also handles that case.
    """
    try:
        os.killpg(process.pid, 0)
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"method": "already_exited", "returncode": process.wait()}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()  # reap our direct child
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return {"method": "sigterm", "returncode": process.wait()}
        except PermissionError:
            # macOS can transiently return EPERM while a signalled child exits.
            # Retry/reap; never treat denied visibility as proof of cleanup.
            pass
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return {"method": "sigkill_after_timeout", "returncode": process.wait(timeout=30)}


@contextlib.contextmanager
def handle_interrupts():
    """Allow finally blocks to run on notebook interrupt / SIGTERM."""
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Experiment interrupted by signal {signum}")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def wait_for_server(process, base_url, log_path, *, timeout=1800, heartbeat=20) -> float:
    started = time.monotonic()
    next_report = started
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}:\n{tail(log_path)}")
        if healthy(base_url):
            # Never accept a different server after our child has died.
            if process.poll() is not None:
                raise RuntimeError("Launched server exited during the health check")
            elapsed = time.monotonic() - started
            progress(f"Server healthy after {elapsed:.1f}s")
            return elapsed
        if time.monotonic() >= next_report:
            lines = tail(log_path, 2000).splitlines()
            last = lines[-1] if lines else "no server output yet"
            progress(f"Loading server: {time.monotonic()-started:.0f}s elapsed; {last[-240:]}")
            next_report = time.monotonic() + heartbeat
        time.sleep(1)
    raise TimeoutError(f"Server startup exceeded {timeout}s.\n{tail(log_path)}")


def run_logged(command, log_path: Path, *, cwd: Path, env: dict,
               stage: str, server=None, timeout=None, heartbeat=20,
               shutdown_timeout=120) -> None:
    """Capture complete output; forward progress and emit a quiet-stage heartbeat."""
    progress(f"Starting {stage}; log: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    lines = queue.Queue()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace", bufsize=1, start_new_session=True,
        )

        def reader():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        next_report = started + heartbeat
        ended = False
        try:
            while not ended or process.poll() is None:
                if server is not None and server.poll() is not None:
                    raise RuntimeError(f"Server exited during {stage} with code {server.returncode}")
                if timeout is not None and time.monotonic() - started > timeout:
                    raise TimeoutError(f"{stage} exceeded {timeout}s; see {log_path}")
                try:
                    line = lines.get(timeout=0.2)
                    if line is None:
                        ended = True
                    else:
                        log.write(line)
                        log.flush()
                        if line.startswith("PROGRESS "):
                            print(line, end="", flush=True)
                except queue.Empty:
                    pass
                if time.monotonic() >= next_report:
                    progress(f"{stage}: {time.monotonic()-started:.0f}s elapsed (still running)")
                    next_report = time.monotonic() + heartbeat
            code = process.wait()
            if code:
                raise RuntimeError(f"{stage} exited with code {code}.\n{tail(log_path, 12000)}")
        finally:
            # Also clean up descendants when their direct parent exited first.
            stop_owned_group(process, timeout=shutdown_timeout)
            thread.join(timeout=2)
            process.stdout.close()
    progress(f"Finished {stage} in {time.monotonic()-started:.1f}s")


def package_results(run_dir: Path, backup_dir: Path | None = None) -> dict:
    """Create a verified local ZIP, then optionally copy it to a mounted backup.

    Called after all measurement processes have stopped, including on failures.
    A failed backup never deletes the verified local archive.
    """
    import zipfile

    run_dir = run_dir.resolve()
    require_local_storage(run_dir)
    archive = run_dir.parent / f"{run_dir.name}_results.zip"
    if archive.exists():
        raise FileExistsError(f"Preserving previous archive: {archive}")
    with tempfile.TemporaryDirectory(prefix="bagel_archive_", dir=run_dir.parent) as temporary:
        built = Path(shutil.make_archive(
            str(Path(temporary) / "results"), "zip",
            root_dir=run_dir.parent, base_dir=run_dir.name,
        ))
        with zipfile.ZipFile(built) as handle:
            bad = handle.testzip()
            if bad is not None:
                raise RuntimeError(f"Archive integrity check failed: {bad}")
        built.replace(archive)
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    receipt = {"local_archive": str(archive), "sha256": digest, "backup_status": "not_requested"}
    progress(f"Verified local ZIP: {archive}")
    if backup_dir is not None:
        try:
            backup_dir = backup_dir.resolve()
            if backup_dir.is_relative_to(Path("/content/drive")) and not os.path.ismount("/content/drive"):
                raise RuntimeError("/content/drive is not mounted; refusing a false Drive backup")
            backup_dir.mkdir(parents=True, exist_ok=True)
            target = backup_dir / archive.name
            if target.exists():
                raise FileExistsError(f"Preserving existing backup: {target}")
            partial = backup_dir / f".{archive.name}.partial.{os.getpid()}"
            if partial.exists():
                raise FileExistsError(f"Preserving unexpected partial backup: {partial}")
            progress("Copying ZIP to backup (measurement has ended)")
            try:
                shutil.copyfile(archive, partial)
                with partial.open("rb") as handle:
                    copied_digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if copied_digest != digest:
                    raise RuntimeError("Backup checksum does not match the local ZIP")
                partial.replace(target)
            finally:
                if partial.exists():
                    partial.unlink()
            receipt.update(backup_status="verified", backup_archive=str(target))
            progress(f"Verified backup: {target}")
        except Exception as exc:
            receipt.update(backup_status="failed", backup_error=f"{type(exc).__name__}: {exc}")
            progress(f"Backup failed; local ZIP is safe. {exc}")
    return receipt
