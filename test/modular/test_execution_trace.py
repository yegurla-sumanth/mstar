import json
import tempfile
import unittest
from pathlib import Path

from mstar.profile.execution_trace import ExecutionTrace


class ExecutionTraceTest(unittest.TestCase):
    def test_disabled_trace_is_a_noop(self):
        trace = ExecutionTrace("worker_0", directory="")
        self.assertFalse(trace.enabled)
        trace.record("ignored", value=1)

    def test_records_clock_and_events_as_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = ExecutionTrace("worker/0", directory=temporary)
            self.assertTrue(trace.enabled)
            self.assertEqual(trace.next_batch_id(), 0)
            self.assertEqual(trace.next_batch_id(), 1)
            trace.record("example", request_handles=[1, 2])
            trace.close()

            paths = list(Path(temporary).glob("*.jsonl"))
            self.assertEqual(len(paths), 1)
            rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
            self.assertEqual([row["event_type"] for row in rows], ["clock_sync", "example"])
            self.assertEqual(rows[1]["request_handles"], [1, 2])
            self.assertIn("perf_counter_ns", rows[1])
            self.assertIn("wall_time_ns", rows[1])


if __name__ == "__main__":
    unittest.main()
