# Bridge training — end‑to‑end walkthrough

Anchored **Stochastic‑Interpolant Bridge** student for the StormCast residual
head (Direction 1 of `flowcast_improvement.md`). This document explains every
moving part of bridge training in this repo: the math, the loss code, the
network, the training loop, the configs, the shell wrapper, and how the bridge
differs from FlowCast.

References to source: [stormcast/utils/bridge_loss.py](stormcast/utils/bridge_loss.py),
[stormcast/utils/trainer_bridge.py](stormcast/utils/trainer_bridge.py),
[stormcast/train_bridge.py](stormcast/train_bridge.py),
[stormcast/utils/nn.py](stormcast/utils/nn.py),
[stormcast/config/bridge.yaml](stormcast/config/bridge.yaml),
[stormcast/config/training/bridge.yaml](stormcast/config/training/bridge.yaml),
[stormcast/config/model/bridge.yaml](stormcast/config/model/bridge.yaml),
[stormcast/zettabyte_scripts/train_bridge.sh](stormcast/zettabyte_scripts/train_bridge.sh).

---

## 1. The idea in one paragraph

FlowCast trains a Conditional Flow Matching velocity field whose ODE
integrates from **white Gaussian noise → standardized residual**
`(M_{t+1} − μ_{t+1}) / σ_data`. The bridge reframes the same generative job
as a stochastic interpolant (Albergo & Vanden‑Eijnden 2023) whose ODE
integrates from **`x_0 = μ_{t+1} + σ_prior·ε` → `x_1 = M_{t+1}` directly**, in
raw physical (log1p‑encoded) units. The frozen regression mean μ is no longer
just a target shift / conditioning channel — it is the **structural prior
endpoint of the flow**. The bridge therefore starts from a sample that is
already physically plausible and only has to traverse the *short* remaining
distance to the truth.

---

## 2. The math

### 2.1 Endpoints and coupling

For each minibatch sample we form a prior endpoint and a data endpoint:

```
x_0 = μ_{t+1} + σ_prior · ε          ε ~ N(0, diag(s_c)²)        (prior)
x_1 = M_{t+1}                                                      (data)
```

`s_c` is the optional per‑channel multiplier `prior_channel_std` (lets qpepre
get a heavier prior than the wind/temperature channels — "Direction 2" of the
improvement plan).

`coupling` picks how the prior‑side noise is paired with targets in the
minibatch:

- `iid` — random pairing (default).
- `ot` — minibatch optimal transport. A Hungarian solve on the spatially
  avg‑pool‑downsampled `(x_0, x_1)` pair returns a permutation `π` that
  re‑indexes the prior noise to be geometrically closest to its target
  ([bridge_loss.py:77](stormcast/utils/bridge_loss.py#L77) — `_ot_permutation`).
  This is OT‑CFM (Tong et al. 2024) applied to the bridge prior.

### 2.2 Interpolant and schedule

Time `t ~ U(t_eps, 1 − t_eps)`. Two schedules are supported, both with
`α(t) = 1 − t`, `β(t) = t`, and an interior noise term `γ(t)·z`, `z ~ N(0,I)`,
that vanishes at the endpoints so the marginals match the prior/data exactly:

| schedule    | γ(t)                       | dγ/dt                  |
|-------------|----------------------------|------------------------|
| `quadratic` | `σ_max · t · (1 − t)`      | `σ_max · (1 − 2t)`     |
| `trig`      | `σ_max · sin(π t)`         | `σ_max · π · cos(π t)` |

Stochastic interpolant and target velocity:

```
x_t = α(t) x_0 + β(t) x_1 + γ(t) z
u_t = dα x_0 + dβ x_1 + dγ z
    = (x_1 − x_0) + σ_max · (1 − 2t) · z      (quadratic case)
```

Setting `σ_max = 0` collapses to a deterministic linear interpolant — the
network then just regresses the residual at every t, useful as an ablation.

### 2.3 Loss

The network `v_θ` is trained to regress against `u_t`; the optimal regressor
is `E[u_t | x_t]`, which is the drift of the marginal probability‑flow ODE
(Albergo 2023, Prop. 2.6). Concretely, with per‑channel weight vector
`β_c = channel_weights`:

```
L_point = mean_{B,c,H,W}  β_c · ( v_θ(x_t, t, condition) − u_t )²
```

### 2.4 Spectral term on qpepre

A radial log‑PSD L1 regularizer is added on the channels listed in
`spectral_channels` (typically `[qpepre]`):

```
x1_pred = x_t + (1 − t) · v_θ(x_t, t, cond)            # 1st-order extrapolation
L_spec  = α_spec · mean_c | log P_k(x1_pred[c]) − log P_k(x1[c]) |
```

This is **exact in the σ_max=0 limit** and a biased‑but‑usable surrogate
otherwise — the bias is `[γ(t) + (1−t)·dγ(t)] · z`, which has zero mean per
pixel and acts as a stochastic regularizer on the log‑PSD term
([bridge_loss.py:295](stormcast/utils/bridge_loss.py#L295)).

`_radial_log_psd` ([bridge_loss.py:52](stormcast/utils/bridge_loss.py#L52))
does an FFT, squares the magnitude, bins by integer `k = round(sqrt(ky² + kx²))`,
averages per bin, and returns log power. The same helper is inlined in
`flowcast_loss.py` so the two losses share an identical spectral term.

### 2.5 Total

```
L = L_point + α_spec · L_spec
```

Returned by `BridgeMatchingLoss.__call__` as a dict
`{loss, pointwise, spectral, sigma_prior, sigma_max}`
([bridge_loss.py:312](stormcast/utils/bridge_loss.py#L312)).

---

## 3. Loss code reference

`BridgeMatchingLoss` lives at
[stormcast/utils/bridge_loss.py:102](stormcast/utils/bridge_loss.py#L102).
Constructor parameters
([bridge_loss.py:148](stormcast/utils/bridge_loss.py#L148)):

| name                | role                                                                                                                      |
|---------------------|---------------------------------------------------------------------------------------------------------------------------|
| `sigma_prior`       | scalar std of the additive Gaussian on x₀; `0` makes the prior endpoint deterministically μ.                              |
| `sigma_max`         | peak of interior γ(t)z; `0` disables the stochastic interpolant (deterministic linear bridge).                            |
| `schedule`          | `quadratic` or `trig`.                                                                                                    |
| `coupling`          | `iid` or `ot` (Hungarian on a downsampled view; deterministic given the batch).                                           |
| `t_eps`             | clamp `t ∈ [t_eps, 1 − t_eps]` so endpoint marginals aren't touched at train time.                                        |
| `channel_weights`   | per‑channel β on the MSE term; must match `kept_HighRes_channels` length.                                                 |
| `spectral_channels` | integer channel indices to add the log‑PSD term on (resolved by name → index in the trainer).                             |
| `spectral_weight`   | α_spec.                                                                                                                   |
| `prior_channel_std` | per‑channel multiplier on the prior noise std; defaults to all‑ones.                                                      |

Important contract: the loss operates in **raw (physical / log1p‑encoded)
space**, not the standardized residual space `FlowCastLoss` uses
([bridge_loss.py:104](stormcast/utils/bridge_loss.py#L104)). Both endpoints
already share the same physical scale, so `sigma_data` does not appear in the
loss. The `sigma_data` config knob is only kept for parity with the EMA
shadow loader.

---

## 4. Network: same backbone, different head semantics

The bridge **reuses FlowCast's `FlowCastPrecond` SongUNet** — same in/out shape,
same conditioning concat, same positional time embedding. The only difference
is at the loss / sampler level: μ is the prior endpoint of the flow, not a
target shift.

This is wired in [stormcast/utils/nn.py:70](stormcast/utils/nn.py#L70):

```python
elif name == "bridge":
    # Shares the FlowCast SongUNet backbone (same in/out shape contract).
    return FlowCastPrecond(
        img_resolution=img_resolution,
        img_channels=target_channels + conditional_channels,
        img_in_channels=target_channels + conditional_channels,
        img_out_channels=target_channels,
        model_type="SongUNet",
        channel_mult=[1, 2, 2, 2, 2],
        attn_resolutions=attn_resolutions,
        additive_pos_embed=spatial_embedding,
    )
```

`sigma_data` and `time_scale` are set on the precond from
`cfg.model.sigma_data` (0.5) and `cfg.model.time_scale` (1000.0)
([trainer_bridge.py:204](stormcast/utils/trainer_bridge.py#L204)). `time_scale`
maps `t ∈ [0,1]` into the SongUNet positional‑embedding range that
originally encoded diffusion timesteps.

---

## 5. Training loop walkthrough

Entry: [stormcast/train_bridge.py](stormcast/train_bridge.py). Hydra picks up
`config/bridge.yaml`, initializes the `DistributedManager`, broadcasts a seed
on rank 0 if `seed < 0`, optionally inits W&B (with `resume=True` if a
`checkpoints_bridge/checkpoint*.pt` exists), and calls
`bridge_training_loop(cfg)`.

The loop lives at [stormcast/utils/trainer_bridge.py:82](stormcast/utils/trainer_bridge.py#L82).
Phase by phase:

### 5.1 Setup ([trainer_bridge.py:82–250](stormcast/utils/trainer_bridge.py#L82))

1. Derive `local_batch_size` and `num_accumulation_rounds` from the global
   `batch_size`.
2. Sanity check: `"regression" ∈ cfg.model.diffusion_conditions` — if missing,
   abort (μ must be available to the loss as the prior endpoint).
3. Set seeds, `cudnn.benchmark`, disable TF32, pick AMP dtype if
   `fp_optimizations` is `amp-bf16`/`amp-fp16`.
4. Load the dataset class via `datasets.dataset_classes[cfg.dataset.name]`,
   build train/valid `InfiniteSampler` + `DataLoader` (drop_last=True,
   pin_memory).
5. Load the **frozen regression net** from `cfg.model.regression_weights`,
   move to device, `.eval().requires_grad_(False)` — the only network whose
   weights never change during this run.
6. Stage the invariants (`lsm, orog`) once per batch.
7. Build the **student** `FlowCastPrecond` (via `name="bridge"`), set
   `sigma_data`, `time_scale`.
8. Build the **EMA shadow** (`ExponentialMovingAverage`, decay 0.999 by
   default) plus a separate `ema_net` shadow module for inference. Both are
   built with the same architecture; `ema.apply_shadow(ema_net)` keeps
   `ema_net` parameters in lock‑step with the EMA running average.
9. Resolve `spectral_channels` from names (e.g. `["qpepre"]`) to indices via
   `state_channels.index(...)`.
10. Construct `BridgeMatchingLoss` from the training config block.
11. Optimizer: `torch.optim.AdamW(lr, betas, weight_decay)` over the student.
12. Wrap student in DDP.
13. **Resume**: `physicsnemo.launch.utils.load_checkpoint` restores
    `student + optimizer + step` from `checkpoints_bridge/`. Then if
    `ema_state.pt` exists, restore EMA state and re‑shadow `ema_net`.

### 5.2 Per‑step body ([trainer_bridge.py:300–388](stormcast/utils/trainer_bridge.py#L300))

For each of `num_accumulation_rounds`:

1. Pull a batch from the iterator → `background`, `state = (S_t pair)`.
2. Build the network inputs with `subtract_regression=False`
   ([nn.py:100](stormcast/utils/nn.py#L100)):
   - `condition = concat(state, μ, invariant)` along the channel dim;
   - `target = M_{t+1}` raw (NOT `M − μ`);
   - `mu` returned separately.
3. Forward through the loss inside AMP autocast:
   - sample ε, optional `_ot_permutation` if `coupling=ot`;
   - sample `t`, compute α/β/γ and derivatives;
   - build `x_t`, compute `u_t`, call `v_pred = student(x_t, t, condition=…)`;
   - per‑channel weighted MSE on `(v_pred − u_t)²`;
   - optional log‑PSD term on `x1_pred = x_t + (1 − t)·v_pred` vs `x_1`;
   - return scalar loss.
4. `loss.backward()`. DDP gradient sync is suppressed for non‑final
   accumulation rounds via `ddp.no_sync()`.

After the accumulation rounds:

5. **Grad clip** at `clip_grad_norm = 1.0`.
6. **LR schedule** — linear warmup over `lr_warmup_steps`, then cosine decay
   into `min_lr = lr · min_lr_ratio` ([trainer_bridge.py:339](stormcast/utils/trainer_bridge.py#L339)).
7. NaN/Inf guard: `torch.nan_to_num` is applied in‑place on every parameter
   gradient before `optimizer.step()` ([trainer_bridge.py:354](stormcast/utils/trainer_bridge.py#L354)).
8. `optimizer.step()`.
9. `ema.update(student, decay=ema_decay)` and `ema.apply_shadow(ema_net)` —
   refresh the inference shadow every step.
10. All‑reduce the scalar `loss / pointwise / spectral` across ranks.
11. Increment `total_steps`; CSV append the smoothed train loss every
    `print_progress_freq` steps.

### 5.3 Validation ([trainer_bridge.py:401–599](stormcast/utils/trainer_bridge.py#L401))

Every `validation_freq` steps:

1. Pull one validation batch, build `condition / target / mu`.
2. Sample with the EMA shadow using `bridge_model_forward`
   ([nn.py:241](stormcast/utils/nn.py#L241)) — the bridge sampler
   returns `M_pred` *directly* (no `+ μ`). See §6.
3. Compute the validation `BridgeMatchingLoss` against the same `target`
   using `student=ema_net`. Note: the validation loss is the **training
   objective** evaluated on the val batch, not a sampled‑output metric. The
   sampled‑output metrics are RMSE/MAE/PS1D, written below.
4. All‑reduce val loss, write to `valid_loss.csv` (rank 0).
5. On rank 0 only, per validation‑plot variable:
   - run `ps1d_plots` to get PSD numerics + figures;
   - accumulate RMSE/MAE between `output_images[i, c]` and `state[1][i, c]`;
   - save `images/{field}/{step}_{i}_{field}.png` (validation heatmap) and
     `..._spec.png` (radial PS);
   - optional NetCDF dump every `output_nc_freq` validations;
   - log to W&B if enabled.
6. Append `rmse_<field>.csv`, `mae_<field>.csv`, `ps1d_<field>.csv`.
7. Replot `loss_curves.png` via `_plot_loss_curves`.

### 5.4 Checkpointing ([trainer_bridge.py:627](stormcast/utils/trainer_bridge.py#L627))

Every `checkpoint_freq` steps (and on the final step), rank 0:

- saves the optimizer + student into `checkpoints_bridge/checkpoint.0.<step>.pt`
  via `physicsnemo.launch.utils.save_checkpoint`;
- saves `ema_state.pt` (the EMA shadow's running average). **`ema_state.pt`
  is what inference loads**, not the optimizer checkpoint.

Resume is automatic: re‑running the same command picks up the latest
`checkpoints_bridge/checkpoint_*.pt` and `ema_state.pt`.

---

## 6. Validation/inference sampler

`bridge_model_forward` at [stormcast/utils/nn.py:241](stormcast/utils/nn.py#L241)
integrates the bridge ODE:

```
ε ~ N(0, diag(prior_channel_std)²)        # or N(0, I) if None
z(0) = μ + sigma_prior · ε
for i in range(num_steps):
    t_i = t_start + i · dt
    if solver == 'midpoint':
        v_half = v_θ(z, t_i,        cond)
        z_half = z + 0.5 · dt · v_half
        v      = v_θ(z_half, t_i + 0.5·dt, cond)
    else:  # 'euler'
        v      = v_θ(z, t_i, cond)
    z = z + v · dt
return z                                # M_pred in raw physical units
```

Defaults: `num_steps=10`, `solver='euler'`, `[t_start, t_end]=[0, 1]`. The
return value is the next‑state field directly — **do not add μ on top**.

Key differences vs. `flowcast_model_forward`:

| concern         | FlowCast                                   | Bridge                                                |
|-----------------|--------------------------------------------|-------------------------------------------------------|
| starting point  | `z(0) ~ N(0, I)` in standardized space     | `z(0) = μ + σ_prior · ε` in raw space                 |
| return rescale  | `z(1) · σ_data` (residual)                 | `z(1)` (full next‑state field)                        |
| user adds μ?    | yes (`M_pred = μ + r`)                     | no                                                    |

---

## 7. Conditioning & target construction

`build_network_condition_and_target` at
[stormcast/utils/nn.py:100](stormcast/utils/nn.py#L100) is shared with FlowCast
and the EDM teacher; the bridge calls it with `subtract_regression=False`
([trainer_bridge.py:317](stormcast/utils/trainer_bridge.py#L317),
[trainer_bridge.py:423](stormcast/utils/trainer_bridge.py#L423)). That flag
flips two things:

- `target` is returned as **raw `M_{t+1}`** (not `M − μ`).
- The regression output μ is still returned as the third tuple element and
  still gets concatenated into `condition` when `"regression" ∈ condition_list`.

So the bridge uses μ **twice**:
1. As one channel in `condition` (so the U‑Net sees it as a contextual input).
2. As the **prior endpoint** that the ODE starts at.

The model config enforces this:
[config/model/bridge.yaml:14](stormcast/config/model/bridge.yaml#L14) has
`diffusion_conditions: ["state", "regression", "invariant"]`. The trainer
also re‑checks the list at startup
([trainer_bridge.py:104](stormcast/utils/trainer_bridge.py#L104)) and raises
if `regression` is missing.

---

## 8. Configs — what each knob does

### 8.1 [`config/bridge.yaml`](stormcast/config/bridge.yaml)

Top‑level Hydra composition. Loads:
- `dataset/era5_rwrf_qpepre` — the cleaned 192×96 Taiwan dataset.
- `model/bridge` — the bridge model block (below).
- `training/bridge` — the bridge training block (below).
- `sampler/edm_deterministic` — kept only for parity (not used at train time).
- `hydra/default`.

Overrides on top:
```yaml
model:
  use_regression_net: True
  regression_weights: ""        # filled in by the shell wrapper
  spatial_pos_embed: True
training:
  loss: 'bridge'
  total_train_steps: 400000
  checkpoint_freq: 5000
  validation_freq: 500
```

### 8.2 [`config/model/bridge.yaml`](stormcast/config/model/bridge.yaml)

- `regression_conditions: [state, background, invariant]` — inputs to the
  frozen regression net (`F_θ(M_t, S_t, I) → μ_{t+1}`).
- `diffusion_conditions: [state, regression, invariant]` — what gets
  concatenated into `condition` for the bridge backbone. `regression`
  **must** be present (enforced by the trainer).
- `attn_resolutions: []` — no self‑attention at any U‑Net stage by default.
- `regression_weights: ""` — path to the frozen `StormCastUNet.0.*.mdlus`.
- `sigma_data: 0.5` — kept for EMA‑shadow / EDM‑side parity only; the bridge
  loss does **not** divide by it.
- `time_scale: 1000.0` — multiplied into `t ∈ [0,1]` before SongUNet's
  positional embedding (keeps the embedding resolution that the original
  diffusion‑timestep input would have had).

### 8.3 [`config/training/bridge.yaml`](stormcast/config/training/bridge.yaml)

Grouped by purpose:

**General**
- `outdir, experiment_name, run_id, rundir` — output paths.
- `num_data_workers: 4`, `cudnn_benchmark: True`.
- `resume_checkpoint: "latest"`, `initial_weights: ""` — auto‑resume or cold
  start from random / a given file.
- `log_to_wandb: False`, `wandb_mode: "online"`.

**Optimization** (FlowCast paper defaults)
- `batch_size: 64`, `batch_size_per_gpu: "auto"`.
- `lr: 5e-4`, `weight_decay: 1e-4`, `adam_betas: [0.9, 0.999]`.
- `lr_warmup_steps: 4000` (≈1% of total), `min_lr_ratio: 0.01`.
- `total_train_steps: 400000`, `clip_grad_norm: 1.0`.
- `loss: 'bridge'`, `fp_optimizations: fp32`, `compile_model: False`.

**Bridge objective**
- `sigma_prior: 0.05`, `sigma_max: 0.5`.
- `schedule: 'quadratic'`, `coupling: 'iid'`, `t_eps: 1e-5`.
- `ema_decay: 0.999`.
- `prior_channel_std: null` (uniform; set per‑channel for the heavier‑tail
  qpepre prior ablation).

**Validation sampler** (diagnostics only)
- `valid_num_steps: 10`, `solver: 'euler'`.

**Hybrid loss**
- `channel_weights: [1.0, 1.0, 1.0, 2.0]` (u10, v10, t2m, qpepre — order
  must match `kept_HighRes_channels`).
- `spectral_channels: ["qpepre"]`, `spectral_weight: 0.1`.

**Validation outputs**
- `validation_plot_variables: ["t2m", "u10", "v10", "qpepre"]`.
- `output_nc: False`, `output_nc_freq: 5`.

---

## 9. Shell wrapper — what the launch script actually sets

[`stormcast/zettabyte_scripts/train_bridge.sh`](stormcast/zettabyte_scripts/train_bridge.sh)
wraps `torchrun --nnodes=1 --nproc_per_node=4 train_bridge.py --config-name bridge`
and overrides the Hydra config for the zettabyte cloud worker:

- **Dataset**: `/workspace/downloads/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026`,
  `HighRes_img_size=[192, 96]`, `kept_HighRes_channels=[u10, v10, t2m, qpepre]`,
  `qpepre_log1p=true`, train `2019/08/01 → 2021/12/31`, valid 2022 full year.
- **Regression checkpoint**: the cleaned‑pipeline regression at step 8000
  (`StormCastUNet.0.8000.mdlus`).
- **Batch / optimizer**: `batch_size=96`, `lr=5e-4`, `lr_warmup_steps=4000`,
  `min_lr_ratio=0.01`, `total_train_steps=400000`, `clip_grad_norm=1.0`,
  `fp_optimizations=fp32`.
- **Bridge knobs**: `sigma_prior=0.05`, `sigma_max=0.5`,
  `schedule=quadratic`, `coupling=iid`, `t_eps=1e-5`, `ema_decay=0.999`,
  `prior_channel_std=[1,1,1,1]` (uniform — flip the last slot for the
  heavier‑tail qpepre experiment).
- **Hybrid loss**: `channel_weights=[1,1,1,2]`, `spectral_channels=[qpepre]`,
  `spectral_weight=0.1`.
- **Validation sampler**: `valid_num_steps=10`, `solver=euler`,
  `validation_freq=250`, `checkpoint_freq=2500`, `print_progress_freq=25`.
- **CUDA**: `CUDA_VISIBLE_DEVICES=0,1,2,3`.

The script also wraps the run in a **log‑capture + auto‑commit** trap
([train_bridge.sh:40–79](stormcast/zettabyte_scripts/train_bridge.sh#L40)):

- `tee` stdout/stderr to `/workspace/Stormcast-Distilation/zettabyte/bridge_train.log`.
- On **any** exit (success, non‑zero, Ctrl‑C, SIGTERM, torchrun crash), the
  trap stages just that one file (with `git add -f` since the path is
  gitignored), commits with a message that records the exit status and a UTC
  timestamp, and pushes. Every git step swallows its own failure so a broken
  remote / hook / auth doesn't lose the local commit. The most recent
  example in `git log`:
  `41cba2a chore(log): bridge_train.log @ 2026-05-26T20:57:55Z (training exit 1)`.

---

## 10. Output layout

After at least one checkpoint:

```
<rundir>/
├── checkpoints_bridge/checkpoint.0.<step>.pt   # student + optimizer
├── ema_state.pt                                # EMA shadow — inference uses this
├── train_loss.csv, valid_loss.csv
├── rmse_{t2m,u10,v10,qpepre}.csv
├── mae_{t2m,u10,v10,qpepre}.csv
├── ps1d_{t2m,u10,v10,qpepre}.csv
├── images/{field}/{step}_{i}_{field}.png       # heatmap
├── images/{field}/{step}_{i}_{field}_spec.png  # radial PS
└── loss_curves.png / loss_curves_live.png
```

For inference, load `ema_state.pt` into a freshly constructed
`FlowCastPrecond` and call `bridge_model_forward(ema_net, condition, mu, …)`.
See [stormcast/inference_bridge.py](stormcast/inference_bridge.py) and
[stormcast/config/bridge_inference.yaml](stormcast/config/bridge_inference.yaml).

---

## 11. How to launch

```bash
# Local (single‑GPU equivalent):
cd stormcast
python train_bridge.py \
    model.regression_weights=/path/to/StormCastUNet.0.8000.mdlus \
    training.experiment_name=my_bridge_run

# Zettabyte cloud (4×GPU, full hyperparameter set, auto‑commit log):
bash stormcast/zettabyte_scripts/train_bridge.sh
```

Re‑run the same command to resume — both `checkpoints_bridge/checkpoint_latest.pt`
and `ema_state.pt` are restored automatically
([trainer_bridge.py:253](stormcast/utils/trainer_bridge.py#L253),
[trainer_bridge.py:277](stormcast/utils/trainer_bridge.py#L277)).

Common Hydra overrides:
```bash
training.batch_size=128
training.lr=2e-4
training.sigma_prior=0.0           # deterministic prior endpoint at μ
training.sigma_max=0.0             # deterministic linear bridge ablation
training.schedule=trig
training.coupling=ot
training.prior_channel_std=[1,1,1,2]   # heavier qpepre prior
training.spectral_weight=0.0       # disable log‑PSD term
training.log_to_wandb=True
```

---

## 12. Bridge vs. FlowCast — direct comparison

| concern                        | FlowCast                                                   | Bridge                                                                |
|--------------------------------|------------------------------------------------------------|-----------------------------------------------------------------------|
| objective family               | I‑CFM (noise → residual)                                   | Stochastic Interpolant (μ → M)                                        |
| training space                 | standardized residual `z = (M − μ)/σ_data`                 | raw (log1p‑encoded) `M`                                               |
| `subtract_regression` flag     | `True` (target = `M − μ`)                                  | `False` (target = `M`)                                                |
| ODE start at sampling          | `z(0) ~ N(0, I)`                                           | `z(0) = μ + σ_prior · ε`                                              |
| ODE end at sampling            | `z(1) · σ_data` → residual, then `M = μ + r`               | `z(1)` is `M_pred` directly                                           |
| extra knobs                    | `sigma_path=0.01` (single)                                 | `sigma_prior`, `sigma_max`, `schedule`, `coupling`, `prior_channel_std` |
| backbone                       | `FlowCastPrecond` SongUNet                                 | **same** `FlowCastPrecond` SongUNet                                   |
| per‑channel β + log‑PSD on qpepre | yes                                                     | **yes (identical)**                                                   |
| ckpt dir                       | `checkpoints_flowcast/`                                    | `checkpoints_bridge/`                                                 |
| inference weights              | `ema_state.pt`                                             | `ema_state.pt`                                                        |
| factory key in `nn.py`         | `"flowcast"`                                               | `"bridge"`                                                            |
| loss class                     | `FlowCastLoss`                                             | `BridgeMatchingLoss`                                                  |
| trainer module                 | `trainer_flowcast`                                         | `trainer_bridge`                                                      |

The hybrid wrapper (`channel_weights`, `spectral_channels`, `spectral_weight`)
is byte‑identical between the two; the difference is the **base flow‑matching
objective and its endpoints**.

---

## 13. Ablation knobs to be aware of

Useful one‑knob ablations the loss / sampler already support:

- `sigma_prior=0` — deterministic prior endpoint at μ; stochasticity comes
  only from γ(t)z. If `sigma_max=0` as well, the bridge collapses to a
  deterministic residual regressor at every t (useful as a sanity baseline).
- `sigma_max=0` — turns off the interior noise term. Spectral surrogate
  becomes **exact** in this limit (no z‑dependent bias).
- `schedule='trig'` — VP‑style higher‑peaked γ; close to I²SB's schedule.
- `coupling='ot'` — minibatch OT‑CFM coupling on the prior noise; pairs each
  ε with the geometrically closest target before computing `x_t` and `u_t`.
- `prior_channel_std=[1,1,1,k>1]` — heavier‑tail prior for the qpepre
  channel; Direction 2 of the improvement plan.
- `spectral_weight=0` — drop the log‑PSD term entirely.

---

## 14. Gotchas

1. **Inference must use `ema_state.pt`.** The optimizer checkpoint is the
   online student; loading it instead of the EMA shadow gives noticeably
   worse samples (same convention as FlowCast).
2. **`bridge_model_forward` returns `M_pred` directly**, not a residual. Do
   not add μ on top of its output.
3. **The validation `loss` CSV is the training objective evaluated on val
   data**, not a sampled‑output metric. RMSE/MAE/PS1D files are the
   sampled‑output metrics.
4. **`regression` must be in `model.diffusion_conditions`.** Both the model
   YAML and the trainer enforce this; without it the loss has no μ to anchor
   to.
5. **`sigma_data` is non‑functional in the bridge loss.** It is kept in
   model.yaml only for compatibility with EDM‑side / EMA shadow loaders.
6. **Channel order matters.** `channel_weights` and `prior_channel_std` are
   positional; they must be ordered to match `kept_HighRes_channels`
   (`[u10, v10, t2m, qpepre]` in the canonical config).
7. **OT coupling needs `B > 1`.** With local batch size 1 the Hungarian step
   is skipped automatically ([bridge_loss.py:258](stormcast/utils/bridge_loss.py#L258)).
8. **`qpepre_log1p=true` at the dataset level is load‑bearing.** Bridge
   training operates in the same encoded space as the regression mean; the
   regression checkpoint at step 8000 was trained on log1p qpepre. Mixing a
   log1p regression mean with a raw‑mm/h bridge target silently corrupts the
   qpepre channel.
