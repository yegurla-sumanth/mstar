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

## 2026-10-16 — clean six-request staggered trace

### Provenance and validity

The first profiling invocation was discarded because a restarted notebook
cell left its original server alive on port 8000. The clean rerun used a new
output directory and port 8017 after verifying that the old process group was
gone. Its source revision was
`88f34273a71d9600707b6d65b5664a59d4d69a44`; the later plotting-backend fix
does not change serving behavior.

- Archived result: `staggered_6_clean_20261016T055925Z.zip`
- Archive SHA-256:
  `8477c0f82328c41f15e7848da0cf7c1d284b525ad279393a69c2bf518ae00a53`
- Hardware: one NVIDIA A100-SXM4-80GB.
- Server ready: 88.061 seconds with cached checkpoint artifacts.
- One warm-up, then six measured requests arriving eight seconds apart.
- Request latencies: 158.015, 203.426, 214.657, 218.403, 214.694, and
  207.153 seconds.
- First measured submission to last completion: 247.143 seconds.
- Shutdown: intentional `SIGTERM`, return code `-15`.

All six requests completed the expected one prompt prefill, 49 LLM refinement
steps, and one VAE decode. The trace contains 89 measured scheduler component
completions representing 306 logical request-component memberships. The plots
completed without trace-validation warnings.

### What the trace shows

Externally sampled GPU utilization averaged 98.99% during the measured window;
95.42% of its 240 one-second samples were at least 95%. Device memory remained
between 51,382 and 51,390 MiB. This is coarse `nvidia-smi` evidence, not proof
that every SM or execution resource was saturated.

Recorded component CUDA durations totaled 245.941 seconds:

- LLM refinement: 244.448 seconds (99.39%).
- Prompt prefill: 0.968 seconds (0.39%).
- VAE decode: 0.526 seconds (0.21%).

The completion-anchored CUDA intervals were effectively back-to-back: 0.320
seconds of positive gap and 0.007 seconds of apparent overlap, the latter small
enough to be timestamp-anchoring noise. This run therefore provides no evidence
of useful component overlap in the current single-worker path. Even ideal,
zero-contention overlap of every measured VAE decode could hide at most its
0.526 seconds of device work, about 0.21% of this workload's makespan. VAE
overlap is not the leading throughput experiment for this configuration.

The stronger scheduling signal is execution granularity. BAGEL's LLM submodule
only declares `decode`, `prefill_text`, and `prefill_vit` batchable; `image_gen`
is excluded. When the scheduler groups multiple unbatchable refinement rows,
the engine uses `_exec_per_request` and executes their full cycles in order
before yielding. The measured group medians follow that implementation almost
perfectly:

| Scheduler group size | Batches | Median refinement CUDA time |
|---:|---:|---:|
| 1 | 14 | 0.827 s |
| 2 | 9 | 1.655 s |
| 3 | 10 | 2.495 s |
| 4 | 9 | 3.326 s |
| 5 | 14 | 4.160 s |
| 6 | 21 | 4.989 s |

Thus a scheduler group of six is not one efficient six-request model forward;
it is approximately six serial 0.83-second forwards forming a nearly
five-second non-preemptive scheduling quantum. Short ready work waits behind
that quantum. Prompt-prefill ready waits across R0–R5 were 0.000, 0.820, 1.647,
2.473, 3.338, and 3.304 seconds. VAE-decode ready waits were 4.157, 3.324,
2.489, 1.659, 0.827, and 0.005 seconds. Scheduled-to-component-start delay was
otherwise only a few milliseconds, locating the delay in scheduler selection
rather than post-selection dispatch.

### Decision and next experiment

Do not implement CUDA-stream overlap next. First isolate the lower-risk
scheduler question: compare the current unbounded scheduler grouping with an
otherwise identical arm that caps `LLM/image_gen` scheduler groups at one.
That arm lets the scheduler reconsider ready prefills and decodes after every
roughly 0.83-second refinement row, without changing the model, request work,
or component priority. If the cap reduces request latency without materially
hurting makespan, a later experiment can separately add boundary-component
priority; do not combine both changes initially.

Use the same frozen six-request arrival trace for at least five paired warmed
repeats, alternate arm order, and keep one loaded server per arm so startup is
outside the comparison. Report request latency, first/last completion,
makespan, ready waits, actual group sizes, total CUDA work, and correctness.
The present single diagnostic run motivates that comparison but is not itself
a performance claim.

## 2026-10-16 — heterogeneous-step B=2 gate completes

The Step 3 paired run completed on an A100-SXM4-80GB at candidate revision
`ff4a90105bf8ffcd25f5554943bf54115c808a4e`. Its execution trace recorded 36
`LLM/image_gen` completions that jointly advanced two requests at different
refinement indices, out of 111 image-component completions. The maximum true
image batch size was two.

Both fixed-seed final images remained visually coherent, but neither was
bitwise identical to its B=1 counterpart. Request 0 had PSNR 46.211 dB, mean
absolute pixel error 0.417, and maximum error 87. Request 1 had PSNR 42.540 dB,
mean absolute pixel error 0.666, and maximum error 172. This is positive
end-result evidence for the tested primitive, not yet single-step latent parity
or a universal numerical tolerance.

The observed B=1 refinement median was about 0.819 seconds. A joint B=2 forward
was about 1.614 seconds, only roughly 1.5% more throughput-efficient than two
B=1 forwards. Request-level continuous batching therefore works, but this run
does not show a large batching gain. It motivates measuring and removing
algebraically inactive CFG branch work.

## 2026-10-16 — Step 4 CFG branch-pruning gate prepared

Branch `feature/bagel-cfg-branch-pruning` adds an explicit, request-overridable
`cfg_branch_policy` while preserving `reference` as the default. The physical
attention/KV plan now carries only the active branch labels. Under default
text-to-image settings, `auto` derives two rows (`main`, `cfg_text`) for steps
0–40 and one `main` row for steps 41–48; the boundary is computed from the
shifted timestep and CFG interval rather than hard-coded.

Local CPU validation passes 18 packing/CFG/policy tests. The three-row helper
delegates exactly to the unchanged reference formula, inactive image-branch
removal is exact in the synthetic formula test, and image editing retains all
three branches when image guidance is active. Python compilation and Git
whitespace checks pass. Full BAGEL import and GPU execution remain unverified
locally because the Mac environment lacks Triton/CUDA.

`notebooks/04_colab_bagel_cfg_branch_gate.ipynb` is pinned to implementation
commit `5b637180a2041e0f39552a9c00a1f7573e44125e`. On one loaded server it runs
`reference`, `drop_image`, `late_main`, `auto`, and an intentionally unguided
`main_only` negative control with identical prompt/seed inputs. It saves
post-step latents after steps 0, 20, 40, 41, and 48, final-image comparisons,
per-step trace timing, and a four-panel plot. Latent dumping synchronizes the
GPU, so this first gate cannot support a speedup claim. Repeated performance
measurement will be a separate no-dump run after numerical review.

## 2026-10-16 — Step 4 A100 failures and eager recovery

The initial A100 gate completed its reference arm but failed on the first
`drop_image` request with `KeyError: 'cfg_img'` under the engine's compiled
LLM forward. Two attempts to isolate the dynamic branch mapping from Dynamo
did not remove that compiled-path failure. These failed runs do not provide
policy correctness or performance results.

A correctness-only retry set `TORCH_COMPILE_DISABLE=1`. It stopped during the
unmeasured reference warm-up, before any policy measurement, with an eager
indexed-assignment error: the combined image sequence was fp32 while the
autocast projection was bf16. The downloaded evidence is
`cfg_orchestrator.json`, `cfg_sweep_client.log`, and `server.log` from
`cfg_branch_gate_b1_eager_correctness_2ca09bfc`; server readiness was 48.052
seconds and shutdown was the orchestrator's intentional SIGTERM.

Revision `751b483336adaac412446ad4ca4c36c701b4991d` makes the storage cast
explicit and makes the engine skip installing compile wrappers when
`TORCH_COMPILE_DISABLE=1`. The clean Step 4 notebook is pinned to that
revision and labels the next run as eager correctness only. Python syntax and
Git whitespace checks pass locally; the new tensor regression test and the
full gate still require the CUDA environment. Compiled performance remains a
separate unresolved gate.
