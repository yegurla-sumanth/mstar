"""Local CPU tests; no model imports, downloads, or GPU launches."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/bagel/scripts"))
import experiment_runtime as runtime


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_drive_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "not Drive"):
            runtime.require_local_storage(Path("/content/drive/MyDrive/logs"))

    def test_symlink_to_drive_rejected(self):
        link = self.root / "linked"
        link.symlink_to("/content/drive/MyDrive")
        with self.assertRaises(ValueError):
            runtime.require_local_storage(link / "logs")

    def test_network_mount_rejected_but_local_overlay_allowed(self):
        mountinfo = self.root / "mountinfo"
        for fs in ("fuse.drive", "nfs4", "cifs"):
            mountinfo.write_text(f"1 0 0:1 / / rw - overlay overlay rw\n2 1 0:2 / {self.root} rw - {fs} test rw\n")
            with self.assertRaises(ValueError):
                runtime.require_local_storage(self.root / "logs", mountinfo=mountinfo)
        mountinfo.write_text("1 0 0:1 / / rw - overlay overlay rw\n")
        runtime.require_local_storage(self.root / "logs", mountinfo=mountinfo)

    def test_busy_port_rejected(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            with self.assertRaisesRegex(RuntimeError, "busy"):
                runtime.require_free_port(sock.getsockname()[1])

    def run_child(self, code, **kwargs):
        runtime.run_logged([sys.executable, "-u", "-c", code], self.root / "child.log",
                           cwd=self.root, env=os.environ.copy(), stage="test child", timeout=5, **kwargs)

    def test_progress_and_full_log(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.run_child("print('PROGRESS pair 1'); print('detail only in log')")
        self.assertIn("PROGRESS pair 1", output.getvalue())
        self.assertNotIn("detail only", output.getvalue())
        self.assertIn("detail only", (self.root / "child.log").read_text())

    def test_failure_includes_real_exception(self):
        with self.assertRaisesRegex(RuntimeError, "KeyError: 'cfg_img'"):
            self.run_child("raise KeyError('cfg_img')")

    def test_timeout_stops_child(self):
        with self.assertRaises(TimeoutError):
            runtime.run_logged([sys.executable, "-u", "-c", "import os,time; print(os.getpid()); time.sleep(20)"],
                               self.root / "timeout.log", cwd=self.root, env=os.environ.copy(),
                               stage="timeout test", timeout=0.5, heartbeat=0.1)
        pid = int((self.root / "timeout.log").read_text().strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_successful_parent_does_not_leave_grandchild(self):
        code = (
            "import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print(p.pid)"
        )
        self.run_child(code)
        pid = int((self.root / "child.log").read_text().strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_owned_group_cleanup(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
        result = runtime.stop_owned_group(child, timeout=2)
        self.assertEqual(result["method"], "sigterm")
        self.assertEqual(result["returncode"], -signal.SIGTERM)
        self.assertEqual(runtime.stop_owned_group(child)["method"], "already_exited")

    def test_dead_server_does_not_pass_health(self):
        child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"], start_new_session=True)
        child.wait()
        with patch.object(runtime, "healthy", return_value=True), self.assertRaisesRegex(RuntimeError, "code 7"):
            runtime.wait_for_server(child, "http://unused", self.root / "missing", timeout=1)

    def test_server_death_stops_client(self):
        child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"], start_new_session=True)
        child.wait()
        with self.assertRaisesRegex(RuntimeError, "Server exited"):
            self.run_child("import time; time.sleep(20)", server=child)

    def test_signal_handler_restored(self):
        original = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(KeyboardInterrupt):
            with runtime.handle_interrupts():
                os.kill(os.getpid(), signal.SIGTERM)
        self.assertEqual(signal.getsignal(signal.SIGTERM), original)

    def test_verified_archive_and_copy(self):
        run = self.root / "run"
        run.mkdir()
        (run / "output.json").write_text('{"status":"failed"}')
        receipt = runtime.package_results(run, self.root / "backup")
        self.assertEqual(receipt["backup_status"], "verified")
        self.assertEqual(Path(receipt["local_archive"]).read_bytes(), Path(receipt["backup_archive"]).read_bytes())
        with zipfile.ZipFile(receipt["local_archive"]) as archive:
            self.assertEqual(json.loads(archive.read("run/output.json"))["status"], "failed")
        with self.assertRaises(FileExistsError):
            runtime.package_results(run)

    def test_backup_failure_preserves_local_archive(self):
        run = self.root / "run"
        run.mkdir()
        with patch.object(runtime.shutil, "copyfile", side_effect=OSError("Drive unavailable")):
            receipt = runtime.package_results(run, self.root / "backup")
        self.assertEqual(receipt["backup_status"], "failed")
        self.assertTrue(Path(receipt["local_archive"]).exists())
        self.assertEqual(list((self.root / "backup").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
