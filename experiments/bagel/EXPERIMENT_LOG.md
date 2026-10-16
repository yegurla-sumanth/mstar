# BAGEL experiment log

## 2026-10-16 — baseline setup begins

### Decision

Start from Sumanth's public fork on branch `bagel-baseline`, based on official
M* revision `7fa8b8e72e26097979805eca8fe40930e088ff64`. Preserve the older
BranchWM checkout state on its original branch.

The first run will use one A100-SXM4-80GB and M*'s single-GPU BAGEL placement.
It retains the model's normal 1024x1024, 50-step, guided image-generation
behavior. It does not introduce concurrency, scheduling changes, quantization,
weight offload, or reduced generation work.

### Prediction

If the environment and loader are correct, the server will become healthy and
a fixed text-to-image request will return a finite, decodable 1024x1024 PNG.
The first request may be slow because it includes lazy compilation.

### Contradicting observation

Installation alone, an HTTP 200, or server health without a valid image does
not satisfy the milestone. A process exit, in-band server error, missing image,
invalid PNG, wrong dimensions, or non-clean shutdown contradicts success.

### Source inspection

- Local M* checkout was clean at `7fa8b8e72e26097979805eca8fe40930e088ff64`.
- Official M* `HEAD` and the user's fork `HEAD` both resolved to that revision.
- Official BAGEL `HEAD` resolved to
  `5019f57d168e5816e8f3f701b17cc816bb7cf24b` without downloading weights.
- Upstream's manual server launcher uses the three-GPU CFG configuration, and
  its Nsight launcher uses a different generic configuration. Neither is used
  unchanged for this single-GPU Colab baseline.
- `BagelModel` previously passed `revision=None` to Hugging Face. The experiment
  branch now threads an explicit `checkpoint_revision` through every config,
  tokenizer, and snapshot lookup. This pins artifacts without changing model
  computation.

### Workload scaling decision

The one-request run is only a smoke test. The first few-request trace will only
debug instrumentation and establish an understandable execution timeline.
Before drawing conclusions, run repeated, controlled workloads at several
arrival patterns; select exact request counts after measuring per-request time
and Colab cost.

### Unresolved

- Whether the candidate Python/PyTorch/FlashInfer stack completes on the current
  Colab image.
- Actual checkpoint download size and cold server-start duration.
- Peak device and host memory for the unchanged single-image request.
- Whether Nsight Systems and CUDA tracing are usable in Colab.

## 2026-10-16 — first Colab request attempt

### Observation

Run `smoke_20261016T043358Z` reached a healthy server and submitted the fixed
image-generation request. During FlashInfer attention planning, its first-use
JIT path tried to execute `ninja` and failed with `FileNotFoundError`. M*
reported the worker failure to the client as HTTP 500. No request profile or
image was produced, so this attempt does not satisfy the baseline milestone.
The orchestrator proceeded through its server-shutdown path.

### Diagnosis and repair

The BAGEL optional dependency set did not request `ninja`; naming it in a
constraints file only pins the package if another requirement selects it. In
addition, invoking a virtual environment's Python by absolute path does not
activate that environment's `bin` directory on `PATH`. The repair adds
`ninja` to the BAGEL dependencies, checks its executable during notebook
setup, and prepends the active interpreter's directory to the server process
`PATH`. The checkpoint cache can be reused for the retry.
