"""Low-overhead JSONL execution tracing for diagnostic scheduling runs.

The trace is disabled unless ``MSTAR_EXECUTION_TRACE_DIR`` is set. Each
worker writes its own file so tracing never adds cross-process locking to the
serving path. Records carry both monotonic and wall-clock timestamps: the
former is used for intervals on one host, while the latter aligns external
GPU telemetry collected by the experiment driver.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class ExecutionTrace:
    """Thread-safe writer owned by one worker process."""

    def __init__(self, worker_id: str, directory: str | None = None):
        if directory is None:
            directory = os.environ.get("MSTAR_EXECUTION_TRACE_DIR")
        self.worker_id = worker_id
        self._lock = threading.Lock()
        self._batch_id = 0
        self._handle = None
        self.path: Path | None = None
        if not directory:
            return

        trace_dir = Path(directory)
        trace_dir.mkdir(parents=True, exist_ok=True)
        safe_worker = "".join(char if char.isalnum() or char in "-_" else "_" for char in worker_id)
        self.path = trace_dir / f"{safe_worker}_{os.getpid()}.jsonl"
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self.record("clock_sync")

    @property
    def enabled(self) -> bool:
        return self._handle is not None

    def next_batch_id(self) -> int:
        with self._lock:
            batch_id = self._batch_id
            self._batch_id += 1
        return batch_id

    def record(self, event_type: str, **fields: Any) -> None:
        if self._handle is None:
            return
        record = {
            "schema_version": 1,
            "event_type": event_type,
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "perf_counter_ns": time.perf_counter_ns(),
            "wall_time_ns": time.time_ns(),
            **fields,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
