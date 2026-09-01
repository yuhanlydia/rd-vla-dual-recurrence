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
BATCH_SIZE=8 EFFECTIVE_BATCH_SIZE=8 bash scripts/run_mikasa_10h.sh
```

After the baseline checkpoint path in `rdvla_mikasa10_dual.yaml` is updated:

```bash
BATCH_SIZE=1 EFFECTIVE_BATCH_SIZE=1 \
  bash scripts/run_mikasa_10h.sh configs/train/rdvla_mikasa10_dual.yaml
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
