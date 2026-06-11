# Model parameter counts — diffusion / flowcast / bridge

How to count the learnable parameters of the three generative residual heads
(and the frozen regression mean), why the numbers come out the way they do, and
a script you can re-run in your training environment.

All numbers below were obtained by **instantiating the real networks and summing
`p.numel()`** — not estimated. See [§5](#5-reproducibility--how-these-numbers-were-produced).

---

## TL;DR

| Network | Factory wrapper | SongUNet in→out | Total params | Trainable |
|---|---|---:|---:|---:|
| **diffusion** | `EDMPrecond` | 14 → 4 | **78,677,380** (~78.7 M) | 78,677,380 |
| **flowcast** | `FlowCastPrecond` | 14 → 4 | **78,677,380** (~78.7 M) | 78,677,380 |
| **bridge** | `FlowCastPrecond` | 14 → 4 | **78,677,380** (~78.7 M) | 78,677,380 |
| regression (frozen) | `StormCastUNet` | 30 → 4 | 78,367,108 (~78.4 M) | 78,367,108 |

**The three generative heads have exactly the same parameter count.** They all
wrap the *same* SongUNet backbone with the same shape contract (14 input
channels = 4 target + 10 conditioning, 4 output channels, `channel_mult=[1,2,2,2,2]`,
`model_channels=128`). `EDMPrecond` and `FlowCastPrecond` differ only in the
*loss/sampler* math (c_skip/c_out vs. velocity field); neither adds learnable
parameters on top of the SongUNet. The bridge is literally `FlowCastPrecond`
again — see [stormcast/utils/nn.py:70-85](stormcast/utils/nn.py#L70-L85).

When training a generative head, the regression net is loaded
`.eval().requires_grad_(False)`, so its ~78.4 M parameters are **not** part of
the head's trainable count — only conditioning input.

---

## 1. How to count — the canonical recipe

The standard PyTorch idiom (this is exactly what was run):

```python
def n_params(module):
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable
```

Drop-in script — save as `stormcast/count_params.py` and run **from the
`stormcast/` directory** (the repo imports `utils.*` as a namespace package, so
cwd must be `stormcast/`):

```python
# stormcast/count_params.py  —  run: cd stormcast && python count_params.py
from utils.nn import get_preconditioned_architecture

def n_params(m):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

# Channel counts for the cleaned Taiwan dataset (4 HighRes target channels).
# generative head conditioning = state(4) + regression(4) + invariant(2) = 10
# regression       conditioning = state(4) + background(24) + invariant(2) = 30
GEN = dict(target_channels=4, conditional_channels=10,
           img_resolution=(192, 96), spatial_embedding=False, attn_resolutions=[])

for name in ["diffusion", "flowcast", "bridge"]:
    net = get_preconditioned_architecture(name=name, **GEN)
    t, tr = n_params(net)
    print(f"{name:11s} total={t:,}  trainable={tr:,}")

reg = get_preconditioned_architecture(
    name="regression", target_channels=4, conditional_channels=30,
    img_resolution=(192, 96), spatial_embedding=False, attn_resolutions=[])
t, tr = n_params(reg)
print(f"{'regression':11s} total={t:,}  trainable={tr:,}")
```

Expected output:

```
diffusion   total=78,677,380  trainable=78,677,380
flowcast    total=78,677,380  trainable=78,677,380
bridge      total=78,677,380  trainable=78,677,380
regression  total=78,367,108  trainable=78,367,108
```

Alternatives: `torchinfo.summary(net, ...)` gives a per-layer table with the
same total; `ptflops`/`fvcore` add FLOP estimates. The `p.numel()` sum is the
ground truth for *parameter count*.

---

## 2. What determines the count

Parameter count is set entirely by the SongUNet hyperparameters and the
**input/output channel counts** — *not* by the image resolution (as long as the
additive positional embedding is off; see [§4](#4-notes--gotchas)).

### 2.1 Backbone hyperparameters (identical across all four nets)

From [stormcast/utils/nn.py:27](stormcast/utils/nn.py#L27):

- `model_type="SongUNet"`, `channel_mult=[1, 2, 2, 2, 2]` → 5 resolution levels,
  base width `model_channels=128` (SongUNet default), `num_blocks=4`.
- `attn_resolutions=[]` → no self-attention anywhere (matches the shipped
  diffusion/flowcast/bridge model configs).
- `spatial_pos_embed=False` in every shipped model config
  ([diffusion.yaml](stormcast/config/model/diffusion.yaml#L26),
  [flowcast.yaml](stormcast/config/model/flowcast.yaml#L30),
  [bridge.yaml](stormcast/config/model/bridge.yaml#L18)).

### 2.2 Channel counts (the only thing that differs)

`target_channels` and `conditional_channels` are computed at runtime in the
trainers — [trainer_flowcast.py:174-190](stormcast/utils/trainer_flowcast.py#L174-L190)
(generative heads) and [trainer.py:387-407](stormcast/utils/trainer.py#L387-L407)
(diffusion/regression). The per-condition channel widths are:

| condition | width | source |
|---|---:|---|
| `state`      | 4  | HighRes channels `[u10, v10, t2m, qpepre]` |
| `background` | 24 | LowRes ERA5 conditioning |
| `regression` | 4  | μ has the same channels as the target |
| `invariant`  | 2  | `lsm, orog` |

The condition *list* is set by `diffusion_conditions` / `regression_conditions`
in the model config:

- **Generative heads** use `diffusion_conditions = ["state", "regression", "invariant"]`
  → `conditional_channels = 4 + 4 + 2 = 10`. The SongUNet input is
  `target (4) + conditioning (10) = 14`, output `= 4`.
  (Note: the 24-channel LowRes background is *not* fed to the generative head — it
  reaches the head indirectly through the regression mean μ.)
- **Regression** uses `regression_conditions = ["state", "background", "invariant"]`
  → `conditional_channels = 4 + 24 + 2 = 30`. SongUNet input `= 30`, output `= 4`,
  and `embedding_type="zero"` (no noise/time embedding).

---

## 3. Results & where the parameters live

Per-section breakdown of the 78,677,380-parameter generative SongUNet
(identical for diffusion / flowcast / bridge):

| Section | Params | Note |
|---|---:|---|
| Decoder (`dec.*`) | 51,636,228 | upsampling path (largest — carries skip concatenations) |
| Encoder (`enc.*`) | 26,712,448 | downsampling path |
| Time embedding (`map_layer0` + `map_layer1`) | 328,704 | sigma/flow-time → feature MLP |
| **Total** | **78,677,380** | |

The full layer-by-layer reconstruction of these three sub-totals — every conv,
norm, and linear, with the formulas — is in
[§6](#6-detailed-layer-by-layer-derivation-how-the-787-m-is-built-up).

### Why the four numbers relate the way they do

The shared convolutional backbone (encoder + decoder, excluding the input conv
and time embedding) is **78,348,676** params. From there:

```
generative head = backbone + time_embed              + input_conv(14ch)
                = 78,348,676 + 328,704                                  = 78,677,380
regression      = backbone + 0 (embedding_type="zero")
                + input_conv(30ch) = 78,348,676 + 18,432 extra weights  = 78,367,108
```

- **Generative − regression = +310,272.** Two effects net out:
  - generative has a time-embedding MLP (`map_layer0`+`map_layer1`) that
    regression lacks: **+328,704**.
  - regression's first conv is wider (30 input channels vs. 14):
    `(30−14) × 128 × 3 × 3 = 18,432` extra weights: **−18,432**.
  - net: `328,704 − 18,432 = 310,272`. ✓
- **diffusion = flowcast = bridge.** Same SongUNet, same 14→4 contract; the
  preconditioner wrappers contribute no learnable parameters.

---

## 4. Notes & gotchas

- **Resolution-independent (as configured).** With `spatial_pos_embed=False`,
  the count is identical at 192×96 and 224×128 — verified. Convolutions are
  spatially agnostic; only the (disabled) additive positional embedding depends
  on H×W.
- **If you enable `spatial_pos_embed=True`**, a learned table of exactly
  `H × W × model_channels` parameters is added: `192×96×128 = 2,359,296` (→
  81,036,676 total) or `224×128×128 = 3,670,016` (→ 82,347,396). The shipped
  configs keep it off.
- **`attn_resolutions`** is empty in all configs. Adding self-attention at any
  level would increase the count; the numbers here assume none.
- **EDM vs. FlowCast preconditioner: 0 parameter difference.** `EDMPrecond`'s
  c_skip/c_in/c_out/c_noise are computed from σ, not learned.
- **EMA shadow is not extra "model" parameters.** FlowCast/bridge keep an EMA
  copy ([ema.py](stormcast/utils/ema.py)) — same 78.7 M shape, doubling
  *memory* but it's a copy of the same parameter set, not new capacity.
- **Memory rule of thumb (fp32).** 78.7 M × 4 bytes ≈ 315 MB of weights;
  AdamW adds two more states per param, so the optimizer footprint is ≈ 3×
  (~945 MB) before activations.

---

## 5. Reproducibility — how these numbers were produced

The numbers were generated by constructing each network via the repo's own
factory `get_preconditioned_architecture`
([stormcast/utils/nn.py:27](stormcast/utils/nn.py#L27)) and summing `p.numel()`.

Environment caveat for *this* workstation: the default interpreter is Python
3.9, but the vendored `physicsnemo/` uses `X | Y` type syntax that is only valid
at runtime under Python **3.10+**, so a bare `import physicsnemo` fails here.
To get the counts, physicsnemo was imported under a Python 3.10+ equivalent.
**Run the [§1](#1-how-to-count--the-canonical-recipe) script in your training
environment** (the cloud worker / wherever `python train_flowcast.py` runs) and
you will reproduce the table exactly.

If you only have the Python 3.9 + torch environment on this box, the import
failure is purely the annotation-syntax incompatibility above — it does not
affect the parameter counts, which are fixed by architecture, not by the
interpreter.

---

## 6. Detailed layer-by-layer derivation (how the 78.7 M is built up)

This rebuilds the **78,677,380** total of the generative SongUNet from scratch.
Every sub-total below was cross-checked against the live `named_parameters()`
shapes, so the arithmetic and the model agree exactly.

The backbone is an NCSN++/DDPM++-style **SongUNet** with base width
`model_channels = 128`, `channel_mult = [1, 2, 2, 2, 2]` (→ 5 resolution
levels), `num_blocks = 4`, time-embedding width `emb = 512`
(`model_channels × 4`), input `14` channels, output `4` channels. There is **no
self-attention at any level** (`attn_resolutions = []`) — *except* one fixed
self-attention block in the decoder bottleneck, which this architecture always
includes regardless of that setting.

### 6.1 Primitive layers (parameter formulas)

For a conv kernel `k = 3` (the default; `1×1` skips use `k = 1`):

| Layer | Parameter count |
|---|---|
| `Conv2d(Cin → Cout, k×k)` | `Cout · (Cin · k² + 1)`  (weights + bias) |
| `GroupNorm(C)` | `2C`  (weight + bias) |
| `Linear(Din → Dout)` | `Dout · (Din + 1)` |

### 6.2 The residual block (`UNetBlock`)

Each block is `GroupNorm → Conv3 → (+ time-embed) → GroupNorm → Conv3`, with a
`1×1` skip projection **only when the channel count changes or the block
resamples** (down/up). With embedding width `E = 512`:

```
P_block(Cin, Cout) = 2·Cin                 # norm0  (GroupNorm on Cin)
                   + Cout·(9·Cin + 1)       # conv0  (3×3, Cin→Cout)
                   + Cout·(E + 1)           # affine (Linear 512→Cout, additive time embed)
                   + 2·Cout                 # norm1  (GroupNorm on Cout)
                   + Cout·(9·Cout + 1)      # conv1  (3×3, Cout→Cout)
                   [+ Cout·(Cin + 1)]       # skip   (1×1, only if Cin≠Cout or resampling)
```

A bottleneck self-attention add-on (channels `C`) costs:

```
P_attn(C) = 2·C            # GroupNorm
          + 3·C·(C + 1)    # qkv  (1×1, C→3C)
          +   C·(C + 1)    # proj (1×1, C→C)
          = 2·C + 4·C·(C + 1)
```

**Worked checks** (match the live shapes exactly):

| Block | Cin→Cout | skip? | compute | params |
|---|---|---|---|---:|
| plain @128 | 128→128 | no | `256 + 128·1153 + 128·513 + 256 + 128·1153` | 361,344 |
| down @128 | 128→128 | yes | `361,344 + 128·129` | 377,856 |
| channel-up | 128→256 | yes | `256 + 256·1153 + 256·513 + 512 + 256·2305 + 256·129` | 1,050,368 |
| plain @256 | 256→256 | no | `512 + 256·2305 + 256·513 + 512 + 256·2305` | 1,312,512 |
| decoder (skip-concat) | 512→256 | yes | `1024 + 256·4609 + 256·513 + 512 + 256·2305 + 256·513` | 2,034,176 |
| attention @256 | — | — | `2·256 + 4·256·257` | 263,680 |

### 6.3 Channel & resolution schedule

`channel_mult = [1,2,2,2,2]` → channels `[128, 256, 256, 256, 256]`. Input
192×96 halves each level: **192 → 96 → 48 → 24 → 12** (block keys are named by
the longer side).

| Level | resolution | channels |
|---|---|---|
| L0 | 192×96 | 128 |
| L1 | 96×48 | 256 |
| L2 | 48×24 | 256 |
| L3 | 24×12 | 256 |
| L4 (bottleneck) | 12×6 | 256 |

### 6.4 Time-embedding MLP — 328,704

Sigma/flow-time → 128-d positional embedding (no params) → two linear layers
(width 512):

```
map_layer0 = Linear(128 → 512) = 512·(128+1) =  66,048
map_layer1 = Linear(512 → 512) = 512·(512+1) = 262,656
                                                ─────────
                                                328,704
```

### 6.5 Encoder — 26,712,448

Each level has `num_blocks = 4` residual blocks; levels L1–L4 are preceded by a
downsampling block. The 128→256 channel lift happens in L1's first block.

| Component | Cin→Cout | each | × | sub-total |
|---|---|---:|---:|---:|
| input conv (3×3, 14→128) | 14→128 | 16,256 | 1 | 16,256 |
| L0 @192 — 4 blocks | 128→128 | 361,344 | 4 | 1,445,376 |
| L1 @96 — down | 128→128 | 377,856 | 1 | 377,856 |
| L1 @96 — block0 (lift) | 128→256 | 1,050,368 | 1 | 1,050,368 |
| L1 @96 — blocks 1-3 | 256→256 | 1,312,512 | 3 | 3,937,536 |
| L2 @48 — down | 256→256 | 1,378,304 | 1 | 1,378,304 |
| L2 @48 — blocks 0-3 | 256→256 | 1,312,512 | 4 | 5,250,048 |
| L3 @24 — down | 256→256 | 1,378,304 | 1 | 1,378,304 |
| L3 @24 — blocks 0-3 | 256→256 | 1,312,512 | 4 | 5,250,048 |
| L4 @12 — down | 256→256 | 1,378,304 | 1 | 1,378,304 |
| L4 @12 — blocks 0-3 | 256→256 | 1,312,512 | 4 | 5,250,048 |
| **Encoder total** | | | | **26,712,448** |

### 6.6 Decoder — 51,636,228

The decoder mirrors the encoder but each block's `conv0` input is **doubled** by
the skip connection concatenated from the encoder (e.g. 256 decoder + 256 skip =
512 in). That doubled `conv0` is why the decoder is ~2× the encoder. The
bottleneck starts with `in0` (residual **+ attention**) and `in1` (residual);
each level above is preceded by an upsampling block; the last two levels taper
256→128, so their skip-concat widths (and thus block sizes) shrink. A final
`GroupNorm + 3×3 conv` projects 128→4 output channels.

| Component | Cin→Cout | each | × | sub-total |
|---|---|---:|---:|---:|
| L4 @12 — in0 (block + attention) | 256→256 | 1,576,192 | 1 | 1,576,192 |
| L4 @12 — in1 (block) | 256→256 | 1,312,512 | 1 | 1,312,512 |
| L4 @12 — blocks 0-4 | 512→256 | 2,034,176 | 5 | 10,170,880 |
| L3 @24 — up | 256→256 | 1,378,304 | 1 | 1,378,304 |
| L3 @24 — blocks 0-4 | 512→256 | 2,034,176 | 5 | 10,170,880 |
| L2 @48 — up | 256→256 | 1,378,304 | 1 | 1,378,304 |
| L2 @48 — blocks 0-4 | 512→256 | 2,034,176 | 5 | 10,170,880 |
| L1 @96 — up | 256→256 | 1,378,304 | 1 | 1,378,304 |
| L1 @96 — blocks 0-3 | 512→256 | 2,034,176 | 4 | 8,136,704 |
| L1 @96 — block4 (taper skip) | 384→256 | 1,706,240 | 1 | 1,706,240 |
| L0 @192 — up | 256→256 | 1,378,304 | 1 | 1,378,304 |
| L0 @192 — block0 | 384→128 | 706,048 | 1 | 706,048 |
| L0 @192 — blocks 1-4 | 256→128 | 541,952 | 4 | 2,167,808 |
| output GroupNorm (128) | — | 256 | 1 | 256 |
| output conv (3×3, 128→4) | 128→4 | 4,612 | 1 | 4,612 |
| **Decoder total** | | | | **51,636,228** |

> The `in0`/`in1` naming is the bottleneck's two middle blocks; `up`/`down` are
> the resampling blocks; `block4` at L1 and `block0` at L0 are smaller because
> their skip-concat width is 384 (256 + the 128-channel top-level skip) rather
> than 512, and L0 outputs only 128 channels.

### 6.7 Grand total

```
   328,704   time-embedding MLP        (§6.4)
26,712,448   encoder                   (§6.5)
51,636,228   decoder                   (§6.6)
──────────
78,677,380   = FlowCast / Bridge / Diffusion SongUNet ✓
```

**Regression** (`StormCastUNet`) reuses the identical encoder + decoder backbone
but (a) drops the time-embedding MLP entirely (`embedding_type="zero"`,
−328,704) and (b) widens only the input conv from 14→128 to 30→128
(`(30−14)·128·9 = +18,432`):

```
78,677,380 − 328,704 + 18,432 = 78,367,108 ✓
```
