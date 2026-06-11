# `experiment_scripts/` — comparison harnesses

Stand-alone post-training experiments that run the trained checkpoints in
[runs/](../runs/) (and the **legacy** old-stormcast checkpoints in
[exp_3_train_2_5_yrs_val_1yr_tp1/](../exp_3_train_2_5_yrs_val_1yr_tp1/))
against a validation set, score the results, and emit plots. Unlike
`stormcast/inference{,_flowcast}.py` (single rollout, single member,
Hydra-driven) these are argparse scripts intended for batch evaluation
across many initial times and ensemble members.

## What lives here

| File | Purpose |
| --- | --- |
| [analysis_plan.md](analysis_plan.md) | The list of metrics and checkpoint pairings the user wants compared. |
| [compare_diffusion_vs_flowcast.py](compare_diffusion_vs_flowcast.py) | Core runner: ensemble rollout for EDM diffusion + (optionally) FlowCast + scoreboard + per-channel PNG panels. Now supports `--kept-channels` and `--skip-flowcast` so the same script handles both the legacy 224×128 dataset and the cleaned 192×96 dataset. |
| [**run_main_experiment.sh**](run_main_experiment.sh) | **Headline 3-way comparison** — legacy old-stormcast (raw qpepre, 224×128) vs. cleaned EDM (log1p) vs. FlowCast (log1p, qpw=2.0), both cleaned legs at the matched ~2 M-sample checkpoint. Runs `compare_diffusion_vs_flowcast.py` twice and stitches the scoreboards. |
| [run_compare_diffusion_vs_flowcast.sh](run_compare_diffusion_vs_flowcast.sh) | Older wrapper around [`log1p_ablation.py`](log1p_ablation.py); the log1p × architecture grid (D1/D2/F1/F2), separate from the main experiment. |
| [**flowcast_nfe_sweep.py**](flowcast_nfe_sweep.py) / [run_flowcast_nfe_sweep.sh](run_flowcast_nfe_sweep.sh) | **NFE Pareto sweep** — hold the FlowCast checkpoint fixed, vary the Euler step count `S ∈ {1, 2, ..., 50}` and record quality (RMSE / CRPS / CSI-M / FSS / HSS / FAR) and wall-clock per sequence. Replicates FlowCast paper Fig. 5 on the Taiwan RWRF domain. Writes `nfe_sweep.{csv,md}` plus `plot_{quality,time,pareto}_vs_nfe.png` for the presentation. |
| [log1p_ablation.py](log1p_ablation.py) / [qpw_ablation.py](qpw_ablation.py) | Dedicated ablation harnesses. See [ablation.md](ablation.md) for the run matrix. |
| [_eval_utils.py](_eval_utils.py) | Shared helpers: dataset wiring, regression + diffusion + flowcast checkpoint loaders, `MetricAccumulator` (RMSE / MAE / bias / CSI tables / radial PSD), `autoregressive_rollout`. |
| [result_table.md](result_table.md) | Latest tabulated single-step skill at a fixed sample-budget cutoff; see CLAUDE.md §6.5 for the headline numbers. |
| `results/<run-tag>/` | Created at runtime: `scoreboard.{md,csv}`, `per_threshold.csv`, `fss_p16.csv`, `rmse_per_channel.csv`, `crps_per_channel.csv`, plus a `panels/` tree with `panels/<channel>/seq{NN}_step{KK}.png` (truth + predictions + prediction − truth diffs, in physical units) and `panels/diff/<channel>/seq{NN}_step{KK}.png` (diff fields only). |

---

## Main experiment — legacy vs. cleaned-2M (run this first)

The thesis headline. Three rows compared on the same 2022 validation year:

| Tag | Method | Dataset | Regression | Generative head | Encoding | Channel order | Grid |
|---|---|---|---|---|---|---|---|
| **A. `legacy_edm`** | Old StormCast (NVIDIA-style residual EDM) | [`exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full`](../exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full) | [`StormCastUNet.0.7500.mdlus`](../exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.7500.mdlus) | [`EDMPrecond.0.70000.mdlus`](../exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus) | raw mm/h | `[t2m, u10, v10, qpepre]` | 224×128 |
| **B. `cleaned_edm`** | New EDM teacher @ ~2 M samples | [`...cleaned_4_27_2026`](../exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026) | [`StormCastUNet.0.8000`](../runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus) | [`EDMPrecond.0.31000`](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.31000.mdlus) | log1p(mm/h) | `[u10, v10, t2m, qpepre]` | 192×96 |
| **C. `cleaned_flow`** | FlowCast student (qpw=2.0, spectral on qpepre) @ ~2 M samples | same as B | same as B | [`FlowCastPrecond.0.20000`](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus) | log1p(mm/h) | `[u10, v10, t2m, qpepre]` | 192×96 |

**Step-to-samples reminder** (for the ~2 M-sample anchor):

- New EDM (`train_diffusion.sh`, batch 64): step 31 250 ≈ 2 M ⇒ pick step
  31 000.
- FlowCast (`train_flowcast.sh`, batch 96): step 20 833 ≈ 2 M ⇒ pick step
  20 000.

Run:

```bash
bash experiment_scripts/run_main_experiment.sh
```

Outputs:

```
experiment_scripts/results/main_experiment/
├── legacy/                                # leg A
│   ├── scoreboard.{md,csv}
│   ├── per_threshold.csv, fss_p16.csv, rmse_per_channel.csv, crps_per_channel.csv
│   └── panels/
│       ├── <channel>/seq{NN}_step{KK}.png       # 2-row: row 1 = truth | diffusion (physical units)
│       │                                        #        row 2 = -      | diffusion − truth   (Δ units, symmetric)
│       └── diff/<channel>/seq{NN}_step{KK}.png  # 1-row: diffusion − truth only
├── cleaned_2M/                            # legs B + C
│   ├── scoreboard.{md,csv}
│   ├── per_threshold.csv, fss_p16.csv, rmse_per_channel.csv, crps_per_channel.csv
│   └── panels/
│       ├── <channel>/seq{NN}_step{KK}.png       # row 1 = truth | diffusion | flowcast
│       │                                        # row 2 = -      | diffusion − truth | flowcast − truth
│       └── diff/<channel>/seq{NN}_step{KK}.png  # diffusion − truth | flowcast − truth only
└── scoreboard_3way.{md,csv}               # final stitched 3-row table
```

Channel-keyed subdirectories (`<channel>` ∈ `{t2m, u10, v10, qpepre}`) make
it trivial to scrape later — e.g. `ls cleaned_2M/panels/qpepre/` gives every
precip panel without grepping filenames. Field rows are plotted in **physical
units** (K, m/s, mm/h — see `CHANNEL_UNITS` in the script, which mirrors
[data_loader_rwrf_era5_stable.py:362](../stormcast/datasets/data_loader_rwrf_era5_stable.py#L362)
where `denormalize_state` returns Kelvin / m/s / mm/h after the inverse log1p
when `qpepre_log1p=True`). Diff rows share a symmetric `RdBu_r` colour scale
centred at 0 with `vmax = 99th-percentile |diff|` so the error structure is
legible even when a model has a tiny mean error.

Env-var overrides:

| Var | Default | Meaning |
|---|---|---|
| `N_SEQUENCES` | 24 | Number of evenly-spaced initial times across 2022. |
| `N_STEPS` | 6 | Autoregressive horizon (hours). |
| `ENSEMBLE` | 4 | Members per sequence (kernel CRPS needs ≥ 2). |
| `DIFFUSION_NFE` | 18 | EDM Heun steps (= 36 NFE). |
| `FLOWCAST_NFE` | 10 | FlowCast Euler steps (= 10 NFE). |
| `N_PANELS_SEQ` | 6 | How many sequences to render PNGs for, per leg. |
| `PANEL_STEPS` | "" (first + last) | Space-separated lead-time indices to render. E.g. `"0 2 5"`. |
| `SEED` | 0 | Base RNG; member `k` uses `seed + 1000·k`. |
| `OUT_DIR` | `experiment_scripts/results/main_experiment` | Where everything lands. |
| `{LEGACY,CLEANED}_{DATA,REG,EDM}`, `CLEANED_FLOW` | (paths in script) | Override individual inputs. |

### Apples-to-oranges caveats

The legacy and cleaned legs **cannot** share one dataset config because:

1. **Grid size.** The legacy zarr is 224×128 (28 672 cells); the cleaned
   zarr is 192×96 (18 432 cells) — a 36 % field-of-view crop. RMSE and
   CRPS are per-pixel means so they're comparable in magnitude, but the
   *fields covered* differ. Footnote when reporting.
2. **qpepre encoding.** Legacy stores raw mm/h; cleaned stores
   log1p(mm/h). Every threshold-based qpepre metric (CSI / FSS / HSS /
   FAR) is computed **after** `denormalize_state` so all qpepre numbers
   land in mm/h regardless — those rows ARE directly comparable.
3. **Channel order.** Legacy checkpoints expect `[t2m, u10, v10, qpepre]`
   (the natural zarr order); zettabyte-trained checkpoints expect
   `[u10, v10, t2m, qpepre]`. `--kept-channels` switches between them.
   The script warns loudly if the loader-returned order disagrees with
   the requested one.

The script picks the same evenly-spaced t0 indices on each dataset
(`np.linspace(0, len-n_steps-1, n_sequences)`), so the two legs evaluate
at the *same dates of 2022* up to a small offset from missing-data
filtering. That's the closest thing to "matched initialisations" you can
have given the dataset gap.

### What to look for in the 3-row scoreboard

- **A → B**: isolates the value of the cleaning pipeline + log1p encoding
  + retraining on the cropped grid. A win for B at matched 2 M samples
  validates the cleaning effort.
- **B → C**: isolates the FlowCast (CFM) generative head vs. the EDM
  diffusion teacher. This is the FlowCast-paper claim re-tested on
  Taiwan RWRF. Expect FlowCast to beat EDM on CRPS and CSI-M while
  matching or slightly trailing on RMSE — see
  [result_table.md](result_table.md) Tables A and C.
- **A → C**: end-to-end win of the new pipeline over the legacy baseline.
  This is the headline number for the thesis.

---

## Other entry points

```bash
# Direct single-config run (one dataset, one cell of the matrix)
python experiment_scripts/compare_diffusion_vs_flowcast.py \
    --n-sequences 32 --n-steps 12 --ensemble 4 \
    --diffusion-checkpoint runs/.../EDMPrecond.0.30000.mdlus \
    --flowcast-checkpoint  runs/.../FlowCastPrecond.0.25000.mdlus \
    --regression-checkpoint runs/.../StormCastUNet.0.8000.mdlus

# Legacy-only (old dataset, old checkpoints, no FlowCast).
# Use the .mdlus archives -- the .pt files at the same step are training-state
# dumps for resume only and Module.from_checkpoint will fail with
# `tarfile.ReadError: not a gzip file` on them.
python experiment_scripts/compare_diffusion_vs_flowcast.py \
    --data-location exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full \
    --hr-size 224 128 --no-qpepre-log1p \
    --kept-channels t2m u10 v10 qpepre \
    --regression-checkpoint exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.7500.mdlus \
    --diffusion-checkpoint  exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus \
    --skip-flowcast \
    --output-dir experiment_scripts/results/legacy_only

# log1p vs raw ablation (orthogonal to the main experiment)
bash experiment_scripts/run_compare_diffusion_vs_flowcast.sh
```

The default dataset (no `--data-location` override) is the cleaned 192×96
zarr at
`exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026/`,
validated over `2022/01/01 → 2022/12/31` (8 759 usable hourly pairs).

---

## Reusable knowledge (read before adding more scripts)

### How to run StormCast / FlowCast inference programmatically

Both pipelines share the same regression mean ``M_t`` plus a residual sampler.
The minimal call chain is:

```python
from physicsnemo.models import Module
from datasets import dataset_classes
from utils.nn import (
    build_network_condition_and_target,
    diffusion_model_forward,    # EDM teacher (wraps physicsnemo deterministic_sampler)
    flowcast_model_forward,     # FlowCast student (Euler / midpoint ODE solver)
)

dataset = dataset_classes["data_loader_rwrf_era5_stable.Dataset"](cfg, train=False)
inv = dataset.get_invariants()
invariant_tensor = torch.from_numpy(inv).to(device).unsqueeze(0)

regression = Module.from_checkpoint(reg_path).to(device).eval()
diffusion  = Module.from_checkpoint(edm_path).to(device).eval()  # EDMPrecond
flowcast   = Module.from_checkpoint(fc_path).to(device).eval()   # FlowCastPrecond

# Per autoregressive step:
condition, _, M_t = build_network_condition_and_target(
    background, [state_pred, state_pred], invariant_tensor,
    regression_net=regression,
    condition_list=("state", "regression", "invariant"),         # diffusion / flowcast
    regression_condition_list=("state", "background", "invariant"),
)
residual = diffusion_model_forward(diffusion, condition, state_pred.shape,
                                   sampler_args=dict(num_steps=18, sigma_min=0.002,
                                                     sigma_max=80.0, rho=7.0, solver="heun"))
# or:
residual = flowcast_model_forward(flowcast, condition, state_pred.shape,
                                  num_steps=10, sigma_data=0.5, solver="euler")
state_pred = M_t + residual    # feed back next step
```

`stormcast/inference.py` and `stormcast/inference_flowcast.py` are the
canonical references for this loop and where to put the de-normalisation /
plotting code. Note `inference.py` uses an older `build_network_condition_and_target`
signature with `lead_time_label`; the version in `utils/nn.py` does not have
that argument — follow `inference_flowcast.py` instead.

### `Module.from_checkpoint` is enough — for both models

`EDMPrecond` and `FlowCastPrecond` are both registered as physicsnemo `Module`
subclasses. The `.mdlus` file produced by `save_checkpoint` includes the
metadata + state, so a plain `Module.from_checkpoint(path)` reconstructs the
network without rebuilding the architecture by hand. The
`stormcast/inference_flowcast.py::_load_flowcast_student` helper exists only
for the special case of loading the **EMA shadow** stored in `ema_state.pt` —
when you want the raw student weights (as the analysis plan does), the
checkpoint loader is one line.

### Channel order and qpepre encoding

- The cleaned zarr stores channels as `[t2m, u10, v10, qpepre]`, but every
  training script in `stormcast/zettabyte_scripts/` overrides
  `dataset.kept_HighRes_channels=[u10, v10, t2m, qpepre]`. The data loader
  re-orders into the canonical training order; **always feed your script
  `kept_HighRes_channels=[u10, v10, t2m, qpepre]`** so checkpoints are
  drop-in compatible with the loader-returned arrays.
- `qpepre` lives in **log1p(mm/h)** on disk. `dataset.denormalize_state`
  applies `expm1` automatically when `dataset.qpepre_log1p=True`. After
  that call all four channels are in physical units (K, m/s, m/s, mm/h),
  so categorical thresholds / RMSE / CRPS can be computed directly on the
  output without any further transform.

### Sampler hyperparameters that must match training

EDM teacher (from `train_diffusion.sh`): `sigma_min=0.002, sigma_max=80.0,
sigma_data=0.5, rho=7.0`. The deterministic sampler defaults to `solver="heun"`,
which doubles the NFE per step (`num_steps=18` -> 36 NFE). Drop to
`solver="euler"` if you want NFE to equal `num_steps`.

FlowCast student (from `train_flowcast.sh` and `config/inference/flowcast.yaml`):
`sigma_data=0.5, num_steps=10, solver="euler"`. `solver="midpoint"` doubles
the NFE.

### Validation set indexing

`Dataset.__len__` returns the number of usable `(t, t+1h)` pairs (8 759 for
2022 in the cleaned dataset because the very last hour has no successor).
For autoregressive sequences pick `t0_indices` that leave room for `n_steps`
extra hours: `t0 <= len(dataset) - n_steps - 1`.

### Metric definitions used in the scoreboard

Quick-reference table; **see [Metrics in detail](#metrics-in-detail) below
for formulas, intuition, edge cases, and how each one is computed.**

All categorical / FSS metrics are computed on the **ensemble mean** of
qpepre in mm/h. CRPS is the kernel CRPS from `physicsnemo.metrics.general.crps.kcrps`
(Zamo & Naveau 2018, unbiased estimator), computed per channel over the
flattened `(K, S*T*H*W)` array and averaged.

| Metric | Definition | Notes |
|---|---|---|
| RMSE | `sqrt(mean((pred_mean - truth)^2))` per channel, averaged over (S*T) | Reported in physical units. |
| CRPS | Kernel CRPS over the K-member ensemble, averaged over space and time | Lower is better; same physical units as the channel. |
| CSI-M | Mean CSI across thresholds `[0.1, 1, 5, 10, 16, 32]` mm/h | Field-wide contingency on qpepre. |
| CSI-P16 | CSI at the single threshold 16 mm/h | Taiwan CWB's hourly heavy-rain bracket. |
| FSS-P16-M | Fractions Skill Score at 16 mm/h, mean over half-widths `[3, 7, 15]` cells | Cleaned grid is ~2 km/cell, so windows ≈ 12 / 30 / 62 km. |
| HSS-M | Mean Heidke Skill Score across the same thresholds as CSI-M | Hit rate corrected for chance. |
| FAR-M | Mean False Alarm Ratio across thresholds | Lower is better. |
| Time/Seq.(s) | Wall-clock per-sequence rollout time, averaged over sequences (single member) | Includes regression mean + sampler + autoregressive feedback. |

---

## Metrics in detail

The scoreboard mixes three families of metric and they are **not directly
comparable** — each one exposes a different failure mode, and a model can
look great on one while losing badly on another. Read this section before
arguing that "model X is better" from a single number.

The notation below uses:
- `K` = ensemble members (`--ensemble`).
- `S` = number of initial times (`--n-sequences`).
- `T` = autoregressive horizon (`--n-steps`).
- `C` = channel count (4: u10, v10, t2m, qpepre).
- `H, W` = grid (192, 96 for the cleaned dataset).
- `pred[k, s, t, c, y, x]` is one ensemble member's prediction at hour
  `t0_s + 1 + t`, in physical units (mm/h for qpepre, K for t2m, m/s for the
  winds). `pred_mean = pred.mean(axis=0)` collapses over `K`.
- `truth[s, t, c, y, x]` is the matching observation.
- All metrics on qpepre are computed **after** the loader's
  `denormalize_state` (so values are mm/h, not standardised log1p).

### 1. RMSE — root mean squared error (deterministic, per channel)

**Formula** — for one channel:
```
RMSE_c = mean_{s,t}  sqrt( mean_{y,x} ( pred_mean[s,t,c,y,x] - truth[s,t,c,y,x] )^2 )
```
i.e. compute spatial RMSE per `(s, t)`, then average over the `S*T` slots.
Code: `per_channel_rmse(pred_mean.reshape(S*T, C, H, W), truth.reshape(S*T, C, H, W)).mean(axis=0)`.

**What it measures** — magnitude of the pixel-wise error of the *ensemble
mean*, in the channel's native units. Penalises both bias and spread.

**Direction / range** — lower is better. Range `[0, +inf)`.

**Intuition** — RMSE rewards smooth, hedged predictions: averaging the
ensemble already shrinks high-frequency disagreement, so a model that
produces sharp but slightly displaced features can have *worse* RMSE than a
blurry one that hits the right neighbourhood. For precipitation in
particular, the ensemble-mean RMSE is dominated by misses on the heavy tail
(qpepre std ≈ 9 mm/h) and is essentially useless as a tool for telling
"sharp and wrong" from "blurry and right". That's why CSI / FSS / CRPS exist.

**This script** — reports RMSE per channel in physical units. We do
not weight by `lsm` or by any rain mask: dry pixels count as legitimate
zero-error contributions for qpepre, which inflates the absolute value but
keeps comparisons across methods consistent.

### 2. CRPS — Continuous Ranked Probability Score (probabilistic)

**Formula** — for one location, the kernel CRPS estimator (Gneiting &
Raftery 2007; unbiased form from Zamo & Naveau 2018, used here with
`biased=False`):
```
                        1   K               1            K  K
CRPS_kernel(X, y)  =   ---  Σ  |X_i - y|  - ---------    Σ  Σ  |X_i - X_j|
                        K  i=1              2 K (K-1)   i=1 j=1
```
where `X = {pred[1..K]}` is the ensemble at that pixel/channel/time and `y`
is the observation. Averaged over all locations / lead times to get a
scalar per channel.

Code: `physicsnemo.metrics.general.crps.kcrps(pred, obs, dim=0, biased=False)`.
We flatten `(K, S, T, H, W)` to `(K, S*T, H, W)` and call kcrps once per
channel.

**What it measures** — the probabilistic generalisation of MAE. The first
term is *skill* (mean distance from forecasts to truth) and the second is
*spread* (mean pairwise distance among forecasts). A perfect ensemble
narrowly centred on the truth gets `CRPS = 0`; a wide ensemble that brackets
the truth pays a spread penalty but earns a small skill term.

**Direction / range** — lower is better, range `[0, +inf)`, same physical
units as the channel.

**Why it needs an ensemble** — `K=1` reduces CRPS to MAE and the spread
term vanishes; you lose all reliability information. The script forces
`--ensemble >= 2` (CRPS is undefined for `K=1` anyway) and the unbiased
estimator we use here is fair across `K` so you can compare runs at
different ensemble sizes without reweighting.

**Calibration sanity check** — if a model has lower CRPS than another at
the same `K` *and* its RMSE on the ensemble mean is comparable, it's both
sharper and better-calibrated. If it wins on CRPS but loses on RMSE, it's
trading mean-error for spread (often the right trade in a probabilistic
context, but flag it).

### 3. CSI — Critical Success Index (deterministic, threshold-based)

**Contingency table** — fix a threshold `tau` (e.g. 16 mm/h) and label each
pixel "wet" if `qpepre >= tau`. Then:
```
TP = #{ pred_wet AND truth_wet }   (hits)
FP = #{ pred_wet AND truth_dry }   (false alarms)
FN = #{ pred_dry AND truth_wet }   (misses)
TN = #{ pred_dry AND truth_dry }
```
**CSI** (also called Threat Score) is:
```
CSI = TP / (TP + FP + FN)
```

**What it measures** — fraction of pixels where the prediction agreed with
the truth on the *positive* class. TN are excluded, so CSI is not gamed by
huge dry areas (which dominate qpepre).

**Direction / range** — higher is better, `[0, 1]`. `CSI=0` means no
overlap; `CSI=1` means every wet pixel was correctly forecast and every
dry pixel was correctly identified.

**Reading the per-threshold breakdown** —
- Low thresholds (`tau=0.1, 1` mm/h) probe overall wet/dry placement.
- Mid (`5, 10` mm/h) probe convective cores.
- High (`16, 32` mm/h) probe extremes; CSI here often crashes because a
  one-pixel displacement in a sharp convective cell counts as one FP +
  one FN.

**This script** —
- `CSI-M = mean over thresholds [0.1, 1, 5, 10, 16, 32] mm/h`. Equal
  weight per threshold.
- `CSI-P16 = CSI at tau=16 mm/h`. Taiwan CWB's hourly heavy-rain bracket
  ("豪雨" classified at 200 mm/24h ≈ 8 mm/h on average; the 16 mm/h bin is
  the closest hourly proxy). Reported separately because it's the
  "operational" precipitation skill the thesis actually cares about.

### 4. FSS — Fractions Skill Score (deterministic, neighbourhood-based)

**Formula** — for threshold `tau` and neighbourhood half-width `w` (so
window size `n = 2w + 1`):
1. Binarise: `O = (truth >= tau)`, `F = (pred >= tau)`.
2. Mean-pool with an `n x n` box (edge-padded): produces fraction maps
   `O_n`, `F_n` in `[0, 1]`.
3. Mean Squared Error of fractions:
   ```
   MSE_n  =  mean_{y,x} ( F_n - O_n )^2
   MSE_ref = mean_{y,x}    F_n^2  +  mean_{y,x}  O_n^2
   ```
4. `FSS = 1 - MSE_n / MSE_ref`.

Code: `_box_pool` uses a summed-area table so cost is `O(H*W)` regardless
of window size.

**What it measures** — agreement between forecast and observed *fraction*
of wet pixels inside each neighbourhood. Tolerates spatial displacements
up to roughly the window size: a convective cell off by one pixel is
nearly perfect at `w=15` even if it's a hard miss at `w=0`.

**Direction / range** — higher is better, `[0, 1]`. `FSS=1` is perfect
agreement; `FSS=0` is the random-forecast baseline (`MSE_n = MSE_ref`).
There is also a "useful" cutoff at `0.5 + 0.5 * f_o` (where `f_o` is the
domain-wide observed wet fraction) — below that, the forecast is worse
than a uniform random one at that threshold.

**This script** —
- We only report `FSS-P16` (`tau = 16 mm/h`) because that's the operational
  threshold; CSI-M already covers the threshold sweep.
- `FSS-P16-M` averages over half-widths `[3, 7, 15]` cells. On the
  cleaned 192x96 grid (≈ 2 km/cell) those windows are ≈ 12 / 30 / 62 km
  diameter — covering a single thunderstorm cell up to a synoptic-scale
  rain band.
- A model that scores well on FSS-P16-M but poorly on CSI-P16 is producing
  the right *amount* of heavy rain in roughly the right area, but mis-locating
  the cores by a few cells. That's a much more interesting failure than
  "no rain at all" and is exactly what FSS was designed to expose.

### 5. HSS — Heidke Skill Score (deterministic, chance-corrected)

**Formula** — using the same contingency table as CSI:
```
N = TP + FP + FN + TN          (total pixels considered)
E = ((TP + FP)(TP + FN) + (FN + TN)(FP + TN)) / N
                                (expected hits + expected correct rejections under independence)
HSS = (TP + TN - E) / (N - E)
```

**What it measures** — fraction of correct categorical predictions
(hits + correct rejections) **above what you'd get by chance** given the
forecast and observation marginal frequencies. Equivalent to Cohen's
kappa for two binary categories.

**Direction / range** — higher is better. Range `(-inf, 1]`:
- `HSS=1`: perfect skill.
- `HSS=0`: forecast is no better than the chance baseline.
- `HSS<0`: forecast is *worse* than the chance baseline (does happen for
  highly biased forecasts at extreme thresholds — flag it).

**HSS vs CSI** — HSS includes the TN term, so on dry-dominated fields it
tends to be much higher than CSI at the same threshold. The chance
correction also makes HSS more honest at thresholds where wet pixels are
rare: a forecast that always predicts "dry" gets `CSI = 0` but
`HSS = 0` too (chance-corrected zero hits = zero skill), whereas accuracy
would be misleadingly high.

**This script** — `HSS-M = mean over [0.1, 1, 5, 10, 16, 32] mm/h`,
matching CSI-M. Per-threshold values dumped to `per_threshold.csv` for
follow-up plotting.

### 6. FAR — False Alarm Ratio (deterministic, threshold-based)

**Formula**:
```
FAR = FP / (TP + FP)
```

**What it measures** — fraction of predicted "wet" pixels that were in
fact dry. Pure precision in the information-retrieval sense, restricted
to the positive class. Note this is **not** the False Alarm Rate
`FP / (FP + TN)` from ROC analysis (a confusing name collision in the
verification literature).

**Direction / range** — lower is better, `[0, 1]`. `FAR=0` means every
predicted wet pixel was a hit; `FAR=1` means every predicted wet pixel
was a false alarm.

**Calibration cross-check** — read FAR alongside CSI and the *frequency
bias* `(TP + FP) / (TP + FN)`:
- low CSI + low FAR + bias < 1 → model is under-forecasting wet pixels
  (conservative; misses dominate).
- low CSI + high FAR + bias > 1 → model is over-forecasting (over-eager;
  false alarms dominate). For sparse, heavy-tailed qpepre this is the
  more common failure for an under-trained student.

**This script** — `FAR-M = mean over [0.1, 1, 5, 10, 16, 32] mm/h`. The
frequency bias is not in the headline scoreboard but is trivial to
compute from `per_threshold.csv` if you want to disambiguate the failure
mode.

### 7. Time/Seq. — wall-clock per-sequence rollout time

**Definition** — the wall-clock time (in seconds) taken to run **one**
autoregressive sequence of length `T` for **one** ensemble member,
averaged across the `S` sequences.

**Code path** — measured around the per-sequence `rollout()` call; we
include the regression-mean forward, the sampler, the autoregressive
feedback, the de-normalisation, and the host transfer of the result. We
**exclude** model loading, dataset construction, and aggregation. On
CUDA we sync after each sequence so the timing reflects work the GPU
actually finished.

**Why one member, not the full ensemble** — Time/Seq. is the latency
metric you want for an operational forecast: how long does a single
forecast take? Multiplying by `K` is mechanical; reporting the unit cost
is more informative.

**Caveats** —
- The *first* sequence absorbs CUDA kernel autotune / cuDNN benchmarking
  overhead. We don't subtract a warm-up: with a typical `--n-sequences=24`
  the first sequence is `1/24` of the average and the bias is small. If
  you need clean numbers for a paper, run with `--n-sequences=1` once to
  warm up, then measure with the larger `S`.
- Heun-18 NFE for the diffusion teacher is **36 model evals** (Heun is
  2nd-order); Euler-18 would be 18. The script labels the NFE in the
  log line so you can compute a normalised "time per NFE" if needed.
- FlowCast `solver=midpoint` doubles the NFE per integration step, same
  caveat.

---

### How to extend the scoreboard

If you add a new metric, follow the same pattern:

1. Add a primitive in [compare_diffusion_vs_flowcast.py](compare_diffusion_vs_flowcast.py)
   that takes `(pred_mean, truth)` (deterministic) or `(preds_ens, truth)`
   (probabilistic) and returns a scalar per channel or per threshold.
2. Wire it into `aggregate(...)` and `write_scoreboard(...)`.
3. If the metric needs per-channel detail, dump it to its own CSV alongside
   `per_threshold.csv` so downstream plotting code can pick it up
   without re-running the rollouts.
4. Document the formula and direction in this section before merging —
   the scoreboard is the single artifact reviewers will look at, and
   having the math next to the table prevents misinterpretation.

### Where the metric implementations live

- Pixel-space deterministic + log-PSD: this script and
  `stormcast/experiments/experiments.py` (more comprehensive cross-method
  variant of the same idea — useful when you want the full distillation grid
  rather than just diffusion vs flowcast).
- Kernel CRPS: `physicsnemo/metrics/general/crps.py::kcrps`.
- Power spectra: `stormcast/utils/spectrum.py::ps1d_plots` (used by
  `experiments.py`).

### Memory / runtime budget

- One sequence, one member, `n_steps=6`, `192x96`, EDM Heun-18 takes
  ~1 s/step on a recent NVIDIA workstation GPU. FlowCast Euler-10 is
  ~3-4x faster.
- Memory for the in-RAM tensor cube `(K, S, T, C, H, W)` with the defaults
  (`K=4, S=24, T=6, C=4, H=192, W=96`) is ≈ 170 MB per method — fine on a
  16 GB host. Bumping `--n-sequences` or `--ensemble` scales linearly.

### Common gotchas

- The shipped configs in `stormcast/config/{inference,dataset}/*.yaml` carry
  NCDR-cluster paths (`/home/master/13/dczy/...`, `/work/jasjou71/...`) and
  the **stale** validation window `2020/09/01 -> 2021/08/31`. Override via
  Hydra (`++dataset.valid_dates=[2022/01/01,2022/12/31]`) or use this
  script's `--valid-dates` flag — never trust the YAML defaults.
- `inference.py` calls `build_network_condition_and_target` with a
  `lead_time_label` kwarg that the function in `utils/nn.py` does not
  accept. Use `inference_flowcast.py` as the reference; it matches the real
  signature.
- The user's analysis plan asked for the **raw `.mdlus` student** for
  FlowCast (not the EMA shadow). That choice is documented in
  `experiments/experiments.py` too — EMA usually scores better, so report
  both if you have time.
- `seed` controls the shared base; member `k` uses `seed + 1000*k` for
  diffusion and `seed + 1 + 1000*k` for FlowCast. Re-running with the same
  seed reproduces the ensemble exactly.

### Runtime environment

The training/inference code expects the `stormcast_env` conda environment
(see `stormcast/zettabyte_scripts/*.sh`). Outside it you'll need at minimum
PyTorch with CUDA, `physicsnemo` (vendored under `physicsnemo/` in this
repo), `xarray`, `zarr`, `omegaconf`, and `matplotlib`. The `run_*.sh`
wrapper activates the env if it exists and otherwise falls through to the
caller's `python`.

1.
exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus  & exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.7500.mdlus (uncropped dataset )
2.  runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus and  runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.20000.mdlus (cleaned dataset)

3. 
runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus and  runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus (cleaned dataset)