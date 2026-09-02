# Dual Recurrence on MIKASA-Robo-VLA

This branch tests one question: are persistent memory across physical time and
recurrent latent reasoning within one control decision complementary?

## Method

RD-VLA resets its scratchpad at every control decision. We add eight persistent
memory tokens and use the updated memory to initialize the next scratchpad:

```text
M_t = U(M_{t-1}, Prelude(o_t), a_{t-1})
S_{t,0} = TruncNormal(0, sigma) + W_M M_t
S_{t,k+1} = R(S_{t,k}, Prelude(o_t), proprio_t)
a_t = C(S_{t,K})
```

`U` has two Transformer updater layers, width 512, eight heads, and FFN width
2048. The updater and projection contain 9,588,224 trainable parameters. The
VLM, Prelude, recurrent core, Coda, and proprio projector are frozen during the
memory phase.

## Experimental design

One dual checkpoint is trained with `K in {1, 2, 4, 8, 12}` and sequence-level
memory dropout 0.3. The same parameters support the factorial interventions:

| condition | memory | recurrence |
|---|---:|---:|
| Reactive | reset | K=1 |
| RD-VLA | reset | K=12 |
| Memory-only | correct | K=1 |
| Dual recurrence | correct | K=12 |

The primary interaction is:

```text
I = (SR(memory,K12) - SR(memory,K1))
  - (SR(reset,K12) - SR(reset,K1))
```

Memory destruction additionally compares correct, reset, episode-shuffled, and
stale memory while holding the current RGB, proprioception, and instruction fixed.
`experiments/robot/mikasa_robo/memory_interventions.py` implements those state-only
replacements, including a cross-episode bank for batch-one evaluation.

## Pilot tasks

The pilot uses 200 training demonstrations per task. The remaining 50 published
demonstrations are reserved for offline diagnostics; closed-loop success rate must
use fresh simulator seeds.

| family | shorter task | long task |
|---|---|---|
| Object | `RememberColor9-VLA-v0` | `RememberColor9-Long-VLA-v0` |
| Sequential | `ChainOfColors5-VLA-v0` | `ChainOfColors5-Long-VLA-v0` |
| Tracking | `ShellGameShuffleTouch-VLA-v0` | `ShellGameShuffleTouch-Long-VLA-v0` |
| Temporal | `BlinkCountButtonPressHard-VLA-v0` | `BlinkCountButtonPressHard-Long-VLA-v0` |
| Delay | `TimedTransferHard-VLA-v0` | `TimedTransferHard-Long-VLA-v0` |

## Setup

The tested machine uses Python 3.10, PyTorch 2.2.0, CUDA 12.1 wheels, and a
24GB RTX 3090. The dependency pins in `pyproject.toml` fix two upstream resolver
problems: RD-VLA otherwise resolves to an incompatible current PyTorch build,
and TensorFlow 2.15 requires compatible protobuf/tensorflow-metadata versions.

```bash
uv sync --python 3.10
git lfs install
```

Download the official RD-VLA initialization:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "hqfang/12_24-24_24_Spatial_40k",
    local_dir="outputs/rdvla_spatial_40k",
)
snapshot_download(
    "Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b",
    local_dir="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
)
PY
```

Download only the ten configured RLDS directories from
`mikasa-robo/mikasa-robo-vla-rlds`. The exact snake-case names are listed in
`configs/train/rdvla_mikasa10_baseline.yaml`.

## Training

The official public RD-VLA checkpoints are LIBERO/CALVIN checkpoints, not
MIKASA checkpoints. First adapt a shared RD-VLA baseline to MIKASA; only then
train the memory module from that baseline.

Probe the largest batch that completes a real forward/backward:

```bash
bash scripts/probe_batch_size.sh
```

Run a wall-clock-bounded baseline. It stops after ten hours at an optimizer-step
boundary and saves a resumable checkpoint:

```bash
BATCH_SIZE=8 EFFECTIVE_BATCH_SIZE=32 \
  bash scripts/run_mikasa_10h.sh configs/train/rdvla_mikasa10_baseline.yaml
```

The wrapper periodically evicts clean MIKASA TFRecord pages with
`POSIX_FADV_DONTNEED` when the host cgroup file cache exceeds 12 GiB. This is
required on machines where the 44 GiB dataset and the trainer share a smaller
memory cgroup; it does not modify or delete dataset files.
It also exports `MALLOC_ARENA_MAX=2` to prevent TensorFlow's many worker
threads from retaining one large glibc allocator arena each.

When baseline training completes, hand off its latest checkpoint automatically:

```bash
bash scripts/run_dual_from_latest_baseline.sh
```

This defaults to the measured frozen-backbone batch 24 and uses one latest
checkpoint directory to avoid filling the experiment disk. Override with
`BATCH_SIZE=24 EFFECTIVE_BATCH_SIZE=24` explicitly if desired.

If a dual stage reaches its wall-time boundary before the campaign is finished,
the continuation supervisor can wait for the current wrapper, find the newest
dual checkpoint, and start another bounded stage (three hours by default):

```bash
setsid nohup bash scripts/continue_dual_after_stage.sh <wrapper-pid> 3 \
  > outputs/mikasa10_dual/continuation.log 2>&1 < /dev/null &
```

Continuation stages restore the checkpoint's optimizer, scheduler, and RNG
state; only the initial baseline-to-dual handoff intentionally starts with a
fresh optimizer.

To guarantee an absolute campaign deadline even if a bounded stage exits
early (for example after a transient data-loader failure), start the takeover
watcher alongside the supervisor. It waits without touching the live trainer,
then resumes from the newest validated checkpoint until the UTC deadline:

```bash
# Replace the timestamp with the campaign's actual UTC deadline.
DEADLINE=$(date -u -d '2026-09-02 07:29:00' +%s)
setsid nohup bash scripts/watch_dual_until_deadline.sh <supervisor-pid> "$DEADLINE" \
  > outputs/mikasa10_dual/deadline_watchdog.log 2>&1 < /dev/null &
```

For one bounded campaign that allocates wall time to both stages, use:

```bash
BASELINE_HOURS=5 DUAL_HOURS=5 scripts/run_mikasa_campaign.sh
```

The wrapper passes each stage's limit through `MAX_WALL_TIME_HOURS` and then
discovers the newest baseline checkpoint before launching dual recurrence.

On an RTX 3090 (23.57 GiB usable), the measured baseline boundary is batch 8:
batch 8 passed consecutive optimization steps at a 19.09 GiB peak, while
batches 10, 12, 16, 24, and 64 reached CUDA OOM. The frozen-backbone
memory-only stage passed batch 24 through a complete 16-step TBPTT update.

The episodic loader materializes only the active episodes as NumPy arrays. The
previous nested TensorFlow step iterator leaked host memory over long runs;
this bounded loader is what makes the batch-8 baseline viable under the shared
44 GiB cgroup. Keep the cache-eviction sidecar enabled for the 44 GiB dataset.

Closed-loop factorial runs append one row per seeded episode and can resume
without duplicating completed conditions:

```bash
bash scripts/run_mikasa_eval.sh \
  --checkpoint outputs/mikasa10_dual/<checkpoint> \
  --output outputs/mikasa_eval/factorial.jsonl \
  --memory correct --k 12
```

The complete five-condition run (including shuffled-memory destruction) is
resumable with:

```bash
bash scripts/run_mikasa_factorial.sh \
  outputs/mikasa10_dual \
  outputs/mikasa_eval/factorial.jsonl
```

It runs `(reset, 1)`, `(reset, 12)`, `(correct, 1)`, `(correct, 12)`, and
`(shuffle, 12)`, then writes the interaction report beside the JSONL. The
evaluator resets
latent state at every episode and implements the official seed stream,
action-chunk FIFO, wrapper stack, and `success_once` latch. MIKASA rendering
requires a working Vulkan-capable NVIDIA driver in addition to CUDA. The
evaluator runs a `vulkaninfo --summary` preflight before loading the checkpoint
and exits with an actionable message when the host exposes only llvmpipe or a
broken ICD; this avoids a late SAPIEN renderer crash after model initialization.
In a Docker/NVIDIA container, request graphics capabilities at container
startup (the training-only default `compute,utility` is insufficient):

```bash
NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
```

The memory phase currently requires gradient accumulation 1 because its loss is
already accumulated across the 16-step environment-time TBPTT window.

## Analysis

Closed-loop evaluation rows use JSONL with at least `task`, `horizon`,
`episode_seed`, `memory`, `k`, and `success`. Compute per-task success rates, the
factorial interaction, paired episode-seed bootstrap intervals, and the preregistered
GO/NO-GO gate with:

```bash
python experiments/robot/mikasa_robo/interaction_analysis.py results.jsonl
```

For the pilot's paired short/long comparison, a task is treated as long when
its official horizon split is `Long` **or** its task ID contains `-Long-`.
MIKASA labels some 600-step `*-Long-VLA-v0` environments as `Medium`, so using
the ID keeps all five intended long families in the preregistered gate.

## Important limitation

With environment-time TBPTT 16, action supervision cannot backpropagate from a
decision hundreds of steps later to the original cue-writing operation. Memory
values persist across detached windows, but gradients do not. A negative result
therefore requires a longer/randomized-TBPTT control before rejecting the memory
hypothesis.

## Implemented checks

`tests/test_persistent_memory.py` verifies state carry, shuffled-memory causal
intervention, and nonzero action-loss gradients into both the updater and memory
projection. End-to-end baseline and two-step dual-recurrence GPU smoke tests have
also completed on the 3090.
