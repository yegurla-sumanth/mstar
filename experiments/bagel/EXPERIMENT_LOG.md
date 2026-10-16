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

## 2026-10-16 — single-image retry succeeds

### Measured result

The retry reused the same Colab runtime and checkpoint cache after installing
`ninja==1.11.1.4` and adding the virtual environment's `bin` directory to
`PATH`. The executed repository revision remained
`db48a7af83adba242dcc277fb89dae5f1a0806e5`; the durable setup repair is in the
subsequent commit `61c43b47c63c6da623045e7a4a7709a93f8eaf05`.

- Result directory:
  `MyDrive/CMU_11868_BAGEL/smoke_20261016T043358Z/single_image_retry_ninja`
- Server ready time: 494.224 seconds.
- Client request boundary: 90.148 seconds.
- Peak externally sampled device memory: 51,770 MiB.
- Image SHA-256:
  `d721d43943b6e13d8062cb91c30a9456805e38db9b81c244de57e5fccc8683f5`.
- Shutdown: intentional `SIGTERM`, return code `-15`.

The smoke client only returns success after receiving exactly one nonempty
image and decoding it as a PNG with the requested dimensions. This satisfies
the technical setup/correctness gate. Sumanth subsequently confirmed that the
saved image was visually coherent, completing the Step 1 baseline gate. The
server-ready and request times include cold loading and first-use compilation
and are not warmed performance results.

## 2026-10-16 — Step 2 staggered-profile design

### Decision

The first profiling run will use the unchanged single-GPU placement with one
unmeasured warm-up followed by six measured requests submitted eight seconds
apart. Each request keeps the normal 1024x1024, 49-refinement-step workload.
This is deliberately larger than the smoke test but remains an instrumentation
debug run, not the sample size for a performance claim.

The opt-in trace records when M* first observes a component as ready, when it
schedules that component, its worker-side service window, its CUDA-event
duration, batch membership, and request order. Client submission/completion
and one-second `nvidia-smi` samples provide aligned request and GPU context.
The analysis notebook produces request, scheduler-component, and GPU-telemetry
plots and retains the raw JSON/JSONL evidence in Drive.

### Prediction

If the current single-worker execution path serializes independent BAGEL
components, the trace will show non-overlapping component CUDA intervals. As
the stagger creates mixed-age requests, at least some ready work may wait while
another request's component executes; those intervals will appear as dashed
ready-to-scheduled spans. If VAE decode and another request's LLM refinement
already overlap—or if ready components consistently schedule without a
meaningful wait—that observation weakens the proposed overlap/scheduling
motivation and must redirect the next experiment.

### Evidence boundary and next gate

Tracing is diagnostic instrumentation and may perturb host scheduling. A
multi-request unbatchable component is timed as one scheduler batch, with its
ordered request IDs retained but without separate sub-request GPU intervals.
The six-request result will validate the trace and expose execution order; it
will not support throughput or tail-latency conclusions. After inspecting it,
choose arrival patterns and a larger paired repeat count based on warmed
latency, memory headroom, and observed variability before implementing any
concurrency mechanism.

### Colab plotting preflight repair

The first Step 2 setup attempt completed dependency installation and both
repository tests, but the combined import probe failed only at Matplotlib.
Separate probes confirmed PyTorch 2.9.1+cu128, an A100-SXM4-80GB CUDA kernel,
and FlashInfer 0.6.14 were healthy. Colab had exported
`MPLBACKEND=module://matplotlib_inline.backend_inline` into the isolated
serving environment, which does not include the notebook-only backend. The
plotting script and preflight now force Matplotlib's noninteractive `Agg`
backend before import; no serving dependency or GPU repair was required.
