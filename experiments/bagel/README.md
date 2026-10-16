# BAGEL single-GPU experiments

This directory contains the reproducible path from an unchanged M* BAGEL
baseline to later execution-timeline experiments. The first milestone is one
valid 1024x1024 text-to-image result on a Colab A100 80 GB. It is a setup and
correctness check, not a performance result.

## Pinned inputs

- M* base revision: `7fa8b8e72e26097979805eca8fe40930e088ff64`
- BAGEL repository: `ByteDance-Seed/BAGEL-7B-MoT`
- BAGEL revision: `5019f57d168e5816e8f3f701b17cc816bb7cf24b`
- Placement: `configs/bagel_single_gpu_pinned.yaml`
- Precision and generation behavior: M* defaults (BF16 weights, 1024x1024,
  50 refinement steps, guidance enabled, no quantization or weight offload)

The dependency constraints are a candidate Colab stack until the first GPU
run succeeds. Every run records the fully resolved environment; successful
versions will then be frozen for subsequent comparisons.

## Colab workflow

Upload `notebooks/01_colab_bagel_smoke.ipynb` to a Colab A100 High-RAM
runtime and run cells in order. The notebook:

1. verifies the actual GPU, host memory, and disk before large downloads;
2. clones Sumanth's fork and records the resolved Git commit;
3. creates an isolated Python 3.12 environment;
4. installs the candidate dependency stack and saves `pip freeze`;
5. downloads the pinned BAGEL snapshot;
6. starts the single-GPU M* server, waits for health, submits one request,
   validates and displays the PNG, and stops the server in a `finally` block;
7. saves logs and metadata under `MyDrive/CMU_11868_BAGEL/`.

The scripts can also be run directly from an allocated Linux/NVIDIA host. Use
`python experiments/bagel/scripts/<script>.py --help` for their interfaces.

After the smoke gate succeeds, run
`notebooks/02_colab_bagel_staggered_profile.ipynb`. It performs one warm-up
request, then submits six measured requests eight seconds apart. The opt-in
worker trace records scheduler-ready, scheduled, component-start, and
component-complete events, including CUDA-event durations. The notebook saves
and displays three aligned plots:

- per-request component and ready-to-scheduled timelines;
- M* component lanes with service windows and CUDA activity;
- externally sampled GPU utilization and total device-memory use.

The trace is enabled only when `MSTAR_EXECUTION_TRACE_DIR` is set by the
profiling orchestrator. It does not add a synchronization beyond the worker's
existing component-completion wait. A multi-request unbatchable component is
represented as one scheduler batch; the ordered `request_ids` field records
its internal per-request execution order, while the plotted interval covers
the full batch rather than separate sub-request GPU intervals.

## Evidence boundary

The smoke run separates installation, server health, request success, and
image validation. Its latency includes cold compilation and is not a headline
measurement. The six-request profiling run is likewise a diagnostic check of
the workload and instrumentation. It can reveal component order, observed
ready-to-scheduled waits, and whether the single worker's recorded CUDA
intervals overlap; it cannot establish throughput, tail latency, or the value
of a scheduling change. Those claims require a larger repeated
characterization with paired runs and controlled arrival patterns.
