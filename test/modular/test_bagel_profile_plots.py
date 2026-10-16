import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class BagelProfilePlotTest(unittest.TestCase):
    def test_synthetic_trace_renders_all_plots(self):
        repo = Path(__file__).resolve().parents[2]
        script = repo / "experiments/bagel/scripts/plot_profile.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_dir = root / "execution_trace"
            trace_dir.mkdir()
            t0 = 1_000_000_000_000
            w0 = 2_000_000_000_000
            requests = []
            trace = []
            for index in range(2):
                request_id = f"bagel-measured-{index:02d}-test"
                submit = t0 + index * 2_000_000_000
                complete = submit + 6_000_000_000
                requests.append(
                    {
                        "status": "success",
                        "request_id": request_id,
                        "request_index": index,
                        "submit_perf_ns": submit,
                        "complete_perf_ns": complete,
                        "client_elapsed_seconds": 6.0,
                    }
                )
                for step, (node, walk) in enumerate(
                    (("LLM", "prefill_text"), ("LLM", "image_gen"), ("vae_decoder", "image_gen"))
                ):
                    start = submit + (step + 1) * 1_000_000_000
                    end = start + 500_000_000
                    trace.append(
                        {
                            "event_type": "batch_scheduled",
                            "worker_id": "worker_0",
                            "pid": 1,
                            "batch_id": index * 3 + step,
                            "node_name": node,
                            "graph_walk": walk,
                            "request_handles": [index],
                            "request_ids": [request_id],
                            "scheduled_ns": start - 100_000_000,
                            "ready_observed_ns": {str(index): start - 300_000_000},
                        }
                    )
                    trace.append(
                        {
                            "event_type": "component_complete",
                            "worker_id": "worker_0",
                            "pid": 1,
                            "batch_id": index * 3 + step,
                            "node_name": node,
                            "graph_walk": walk,
                            "request_ids": [request_id],
                            "batch_size": 1,
                            "component_start_ns": start,
                            "gpu_complete_ns": end,
                            "cuda_duration_ms": 400.0,
                        }
                    )
            (root / "profile_summary.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "measurement_start_perf_ns": t0,
                        "measurement_start_wall_ns": w0,
                        "measured_requests": requests,
                    }
                )
            )
            (trace_dir / "worker_0_1.jsonl").write_text("\n".join(json.dumps(row) for row in trace) + "\n")
            samples = [
                {
                    "sample_time_ns": w0 + index * 1_000_000_000,
                    "utilization_gpu_percent": 10 + index * 5,
                    "memory_used_mib": 50_000 + index * 10,
                }
                for index in range(10)
            ]
            (root / "gpu_samples.json").write_text(json.dumps({"samples": samples}))
            output = root / "plots"
            env = os.environ.copy()
            env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input-dir",
                    str(root),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                env=env,
                text=True,
            )
            for filename in (
                "request_component_timeline.png",
                "scheduler_component_timeline.png",
                "gpu_telemetry.png",
                "plot_manifest.json",
            ):
                path = output / filename
                self.assertTrue(path.exists(), filename)
                self.assertGreater(path.stat().st_size, 0, filename)


if __name__ == "__main__":
    unittest.main()
