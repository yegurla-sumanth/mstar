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

## Evidence boundary

The smoke run separates installation, server health, request success, and
image validation. Its latency includes cold compilation and is not a headline
measurement. A later profiling run will use a few staggered requests to debug
the timeline, followed by a larger repeated characterization before any
concurrency or scheduling conclusion.
