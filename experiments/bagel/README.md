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
and displays four plots:

- per-request component and ready-to-scheduled timelines;
- M* component lanes with service windows and CUDA activity;
- externally sampled GPU utilization and total device-memory use;
- refinement CUDA time by scheduler group size beside per-request prompt and
  VAE ready waits, separating true model batching from scheduler grouping.

The experimental Step 3 correctness gate is
`notebooks/03_colab_bagel_heterogeneous_batch_parity.ipynb`. It checks out the
immutable prototype commit `ff4a90105bf8ffcd25f5554943bf54115c808a4e`, runs
paired serial B=1 and heterogeneous B=2 profiles with two staggered requests,
compares their final images, and requires trace evidence that a real
multi-request `LLM/image_gen` forward contained requests at different loop
indices. Passing this gate establishes only the tested batching primitive's
GPU correctness evidence; it is not a throughput or scheduling-policy result.

That gate completed on one A100-SXM4-80GB. The trace recorded 36 true
multi-request heterogeneous-step image completions with maximum image batch
size two. The two final-image comparisons had PSNR 46.21 dB and 42.54 dB;
neither was bitwise identical. The observed B=2 refinement forward was only
about 1.5% more throughput-efficient than two B=1 forwards, so branch-level
work inside each request is the next performance target rather than a claim
that request batching alone is sufficient.

Step 4 is `notebooks/04_colab_bagel_cfg_branch_gate.ipynb`. It checks out the
immutable implementation commit `eb7327801785202639cdbdb9d63e5dad265f3290`
and compares five request-selectable policies on one loaded server:

- `reference`: unchanged three CFG rows on every refinement step;
- `drop_image`: remove only an inactive image-guidance row;
- `late_main`: keep the reference rows inside the CFG interval, then use main;
- `auto`: use two rows through step 40 and main only for steps 41–48 under the
  default text-to-image settings;
- `main_only`: intentionally unguided negative control.

The gate saves post-step latents at steps 0, 20, 40, 41, and 48, final images,
execution traces, and a four-panel plot. Those latent dumps synchronize the GPU,
so its timings diagnose potential only. Repeated timing runs must disable latent
capture and follow only after the numerical gate is reviewed.

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
