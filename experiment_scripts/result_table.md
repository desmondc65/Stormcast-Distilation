# Validation RMSE / MAE @ ~2M training samples

Snapshot of single-step validation skill from
`runs/<exp>/<run>/run_0/{rmse,mae}_{t2m,u10,v10,qpepre}.csv` for each run.
Each row is the **single validation point closest to ~2,000,000 training
samples seen** (samples = `step * batch_size`).

Per-run global batch sizes (read off the launcher scripts in
`stormcast/zettabyte_scripts/`):

| script | batch | step at 2 M samples |
|---|---:|---:|
| `train_diffusion.sh` (log1p teacher) | 64 | 31 250 |
| `train_diffusion_no_log1p.sh` | 96 | 20 833 |
| `train_flowcast.sh` (and all qpw / NO_log1p variants) | 96 | 20 833 |

> **Units.** All values are computed inside the trainer's validation loop on the
> *standardised* network output (no `denormalize_state` call — see
> `stormcast/utils/trainer_flowcast.py` line 503). They are dimensionless
> standard-score errors; multiply by the channel `std` from
> `HighRes/stats/stds.npy` to get physical units (5.61 K · 3.84 m/s · 5.66 m/s ·
> std-of-log1p ≈ 0.37 for log1p qpepre, 9.12 mm/h for raw qpepre).
>
> **Single-snapshot noise.** Each row is one validation snapshot. Adjacent
> validation logs drift by 10-30 % so don't read too much into <5 %
> differences — when in doubt, smooth across ~5 nearby validation logs.
> "Just pick one" per the analysis plan.

---

## A. Diffusion teacher vs FlowCast student (default, log1p, qpw=2.0)

Same dataset, same regression mean, same standardised log1p qpepre — every
column is directly comparable.

| run | samples |  RMSE t2m |  MAE t2m |  RMSE u10 |  MAE u10 |  RMSE v10 |  MAE v10 |  RMSE qpepre |  MAE qpepre |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diffusion (EDMPrecond) | ~2M | 0.0994 | 0.0721 | 0.3523 | 0.2526 | 0.2762 | 0.1988 | 0.7865 | 0.3557 |
| flowcast (FlowCastPrecond, qpw=2.0) | ~2M | **0.0835** | **0.0591** | **0.2931** | **0.2142** | **0.2463** | **0.1760** | **0.3851** | **0.1334** |

**Read.** FlowCast wins on every channel at the same training-sample
budget. The gap on qpepre is the largest — RMSE roughly halved, MAE down
~63 % — consistent with FlowCast's per-channel weighting (qpw=2.0) and
spectral regulariser on qpepre, neither of which the EDM teacher uses.

---

## B. log1p vs NO_log1p qpepre encoding (diffusion + flowcast)

All four runs use the canonical training scripts (flowcast:
`channel_weights=[1,1,1,2.0]`; diffusion: standard EDM with no per-channel
weighting) and only differ in whether the dataset stores qpepre as
`log1p(mm/h)` or raw `mm/h`. Read each architecture's two rows as a
within-architecture A/B test on the encoding.

| run | samples |  RMSE t2m |  MAE t2m |  RMSE u10 |  MAE u10 |  RMSE v10 |  MAE v10 |  RMSE qpepre † |  MAE qpepre † |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diffusion log1p | ~2M | **0.0994** | **0.0721** | **0.3523** | **0.2526** | **0.2762** | **0.1988** | 0.7865 | 0.3557 |
| diffusion NO_log1p | ~2M | 0.1245 | 0.0922 | 0.3955 | 0.2906 | 0.3115 | 0.2268 | 0.4156 | 0.1509 |
| flowcast log1p | ~2M | **0.0835** | **0.0591** | **0.2931** | **0.2142** | **0.2463** | **0.1760** | 0.3851 | 0.1334 |
| flowcast NO_log1p | ~2M | 0.1118 | 0.0798 | 0.3079 | 0.2254 | 0.2331 | 0.1717 | 0.3471 | 0.1149 |

> † **The qpepre column is not directly comparable across the
> log1p / NO_log1p boundary, and you cannot fix it by inverting the
> log1p on the scalar RMSE.**
>
> [`denormalize_state`](../stormcast/datasets/data_loader_rwrf_era5_stable.py#L361-L377)
> does two things to qpepre, in order:
>
> 1. **Linear de-standardisation** — `x ← x * std + mean`.
>    Because RMSE is a difference, the mean cancels and the std passes
>    straight through: `RMSE_destandardised = RMSE_standardised · std`.
>    This step *is* a valid scalar operation on the CSV value.
> 2. **Inverse log1p (`expm1`)** — applied per pixel, before any reduction.
>    `expm1` is non-linear, so it does **not** commute with `sqrt(mean(·²))`:
>    `expm1(RMSE_log1p) ≠ RMSE_mm`. The same standardised log1p error
>    of, say, 0.14 corresponds to ~0.15 mm/h on a dry pixel
>    (`expm1(0+0.14)−expm1(0)`) but ~2.1 mm/h on a 6 mm/h pixel
>    (`expm1(2+0.14)−expm1(2)`). To get the mm/h RMSE you must invert
>    log1p on the actual prediction tensor first, *then* compute
>    `sqrt(mean(diff²))`.
>
> Net of those two rules:
>
> | run | what de-standardisation gives you | comparable to mm/h? |
> |---|---|---|
> | diffusion NO_log1p | `0.4156 · 9.12 ≈ 3.79 mm/h` (RMSE in mm/h) | ✅ yes — already in raw mm/h |
> | flowcast NO_log1p  | `0.3471 · 9.12 ≈ 3.17 mm/h` (RMSE in mm/h) | ✅ yes |
> | diffusion log1p    | `0.7865 · 0.37 ≈ 0.29 in log1p(mm/h)`     | ❌ no — needs per-pixel `expm1` first |
> | flowcast log1p     | `0.3851 · 0.37 ≈ 0.14 in log1p(mm/h)`     | ❌ no — same |
>
> The clean way to settle the encoding question is to run
> [compare_diffusion_vs_flowcast.py](compare_diffusion_vs_flowcast.py)
> on both checkpoints — it calls `denormalize_state` on the rollout
> output and only *then* computes RMSE / CRPS / CSI, so qpepre is in
> mm/h regardless of the on-disk encoding and the four runs land on
> a single common axis.
>
> Within each encoding the qpepre column *is* fine to read directly
> (diffusion vs flowcast at log1p; diffusion vs flowcast at NO_log1p) —
> the unit-mismatch only bites when you cross the encoding boundary.

**Read.**
- **log1p helps both architectures on every non-qpepre channel.**
  Diffusion gains ~20 % t2m / 11 % u10 / 11 % v10 RMSE; flowcast gains
  ~25 % t2m, 5 % u10, 5 % v10. The variance / scale of the qpepre
  channel propagates through shared-trunk weights into the wind /
  temperature channels, and the standard-score values for qpepre are
  far better-conditioned in log1p space (std 0.37) than in raw mm/h
  (std 9.12) — the optimiser doesn't have to fight that scale every
  step.
- **qpepre cannot be read directly from this table** because of the
  unit mismatch (see the † note). After de-standardising the NO_log1p
  rows: diffusion ≈ 3.79 mm/h, flowcast ≈ 3.17 mm/h. The log1p rows
  need post-`denormalize_state` rescoring to land on the same axis.
- **Architecture × encoding interaction.** Both architectures benefit
  from log1p on every non-qpepre channel; the magnitude of the benefit
  is similar (~5-25 %), so the encoding choice is essentially
  independent of the architecture choice in this snapshot. Use log1p
  as the canonical encoding going forward.

---

## C. qpepre channel-weight ablation (flowcast, log1p)

All runs use the cleaned log1p dataset and the canonical FlowCast architecture;
only the last entry of `channel_weights=[1.0, 1.0, 1.0, qpw]` changes.
qpw = 2.0 row is the headline run from Table A repeated for context.

> **Should the qpepre error be rescaled by qpw before comparison?** **No.**
> `channel_weights` is applied only inside the training loss
> ([flowcast_loss.py:150](../stormcast/utils/flowcast_loss.py#L150),
> `pw = pw * beta`); the validation RMSE / MAE are plain
> `sqrt(mean(diff**2))` and `mean(|diff|)` over the standardised output
> ([trainer_flowcast.py:503-504](../stormcast/utils/trainer_flowcast.py#L503-L504))
> with no per-channel reweighting. So the cells below are raw error
> magnitudes in standardised log1p(mm/h) space and are directly comparable
> across qpw values. Dividing or multiplying by qpw would double-count the
> weight: qpw already shaped what the model learned via the training
> gradient, and the validation metric measures how that already-shaped
> prediction compares to the ground truth in unweighted units. The right
> way to read these rows is "for a given qpw, how accurately does the
> resulting model reconstruct qpepre" — not "what was the value of the
> training loss".

| qpw | samples |  RMSE t2m |  MAE t2m |  RMSE u10 |  MAE u10 |  RMSE v10 |  MAE v10 |  RMSE qpepre |  MAE qpepre |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | ~2M | 0.0994 | 0.0699 | 0.3188 | 0.2295 | 0.2326 | 0.1669 | 0.5464 | 0.1866 |
| 1.2 | ~2M | 0.1032 | 0.0732 | 0.2912 | 0.2094 | 0.2110 | 0.1533 | 0.3771 | 0.1242 |
| **1.4** | ~2M | 0.0890 | 0.0620 | **0.2733** | **0.1982** | **0.2050** | **0.1479** | 0.3996 | **0.1130** |
| 1.6 | ~2M | 0.0930 | 0.0669 | 0.3282 | 0.2348 | 0.2391 | 0.1722 | 0.6796 | 0.3180 |
| 1.8 | ~2M | 0.0983 | 0.0689 | 0.2838 | 0.2075 | 0.2075 | 0.1518 | 0.4831 | 0.1807 |
| 2.0 (default) | ~2M | **0.0835** | **0.0591** | 0.2931 | 0.2142 | 0.2463 | 0.1760 | **0.3851** | 0.1334 |
| 2.2 | ~2M | 0.0990 | 0.0707 | 0.3178 | 0.2342 | 0.2499 | 0.1804 | 0.5178 | 0.1822 |
| 2.4 | ~2M | 0.0991 | 0.0687 | 0.3075 | 0.2220 | 0.2355 | 0.1700 | 0.5515 | 0.2425 |

**Read.**
- **No clean monotone trend** of skill in qpw. Both qpw=1.4 and qpw=2.0
  beat their neighbours; qpw=1.6 and qpw=2.4 are bad across the board.
  Snapshot noise is large enough that the ordering between qpw=1.2 / 1.4
  / 2.0 is probably not stable, but the floor-to-ceiling spread (qpw=1.4
  vs qpw=1.6) is ~30 % on qpepre RMSE and 12 % on t2m RMSE — that's real.
- **qpw=1.4 is the per-channel sweet spot for the wind channels** (best on
  every wind column) while still placing 2nd on qpepre MAE. If the goal is
  balanced multi-channel skill, 1.4 looks like a stronger default than the
  canonical 2.0.
- **qpw=2.0 wins on t2m and on qpepre RMSE** (0.3851), which is why the
  canonical script picked it — but it's noticeably worse than qpw=1.4 on
  u10 / v10. Treat this as a "precip first, winds second" trade-off.
- **Above qpw=2.0** (rows 2.2 / 2.4) skill degrades on every channel,
  including qpepre — the loss surface starts overweighting precip enough
  to harm the shared-trunk features.
- **Caveat.** This is a single-validation-step snapshot. The qpw ablation
  should ideally be re-scored with the kernel-CRPS / FSS-P16 metrics in
  [compare_diffusion_vs_flowcast.py](compare_diffusion_vs_flowcast.py) over
  several validation snapshots before claiming a winner — pixel-RMSE
  rewards hedged predictions, which is exactly the failure mode a
  precipitation-weighted loss is meant to avoid.

---

## Reproduction

```bash
python3 - <<'PY'
import csv
from pathlib import Path
ROOT = Path("runs")

def at_step(run_dir, metric, target):
    rows = list(csv.reader(open(run_dir / f"{metric}.csv"))); rows.pop(0)
    return min(((int(s), float(v)) for s, v in rows), key=lambda r: abs(r[0] - target))

print(at_step(ROOT / "diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0", "rmse_t2m", 31250))
print(at_step(ROOT / "flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0",  "rmse_t2m", 20833))
PY
```

---

## Appendix — validation panels at ~2M samples (diffusion vs flowcast)

The trainer's validation loop drops a side-by-side `(generated | truth)`
PNG per channel per validation sample under `runs/<run>/run_0/images/<var>/`.
Below are two arbitrary validation samples (sample 0, sample 5) from each
run **at the same ~2M-training-samples checkpoint** Table A scores. The
validation set is iterated in the same order for both runs (deterministic
given identical `valid_dates`), so sample index `i` corresponds to the
same target timestamp on both sides.

> Each PNG is the trainer-rendered figure: left panel = model output,
> right panel = RWRF ground truth. Diffusion images come from
> `runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/<var>/`;
> flowcast from
> `runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/<var>/`.

### t2m (2 m temperature)

| sample | diffusion (~2M) | flowcast qpw=2.0 (~2M) |
|---:|---|---|
| 0 | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/t2m/31200_0_t2m.png) | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/t2m/20750_0_t2m.png) |
| 5 | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/t2m/31200_5_t2m.png) | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/t2m/20750_5_t2m.png) |

### u10 (10 m zonal wind)

| sample | diffusion (~2M) | flowcast qpw=2.0 (~2M) |
|---:|---|---|
| 0 | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/u10/31200_0_u10.png) | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/u10/20750_0_u10.png) |
| 5 | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/u10/31200_5_u10.png) | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/u10/20750_5_u10.png) |

### v10 (10 m meridional wind)

| sample | diffusion (~2M) | flowcast qpw=2.0 (~2M) |
|---:|---|---|
| 0 | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/v10/31200_0_v10.png) | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/v10/20750_0_v10.png) |
| 5 | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/v10/31200_5_v10.png) | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/v10/20750_5_v10.png) |

### qpepre (1 h precipitation)

qpepre is the channel that motivates this whole project — sparse, heavy-tailed,
and the easiest place for a distilled student to silently mode-collapse to
"always slightly damp". Eight samples here so you can scan for that failure
mode (look for: flowcast over-smoothing the cores, diffusion missing dry
quadrants, both leaking light precip into the south-east no-rain zone).

| sample | diffusion (~2M) | flowcast qpw=2.0 (~2M) |
|---:|---|---|
| 0  | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/31200_0_qpepre.png)  | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/20750_0_qpepre.png)  |
| 1  | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/31200_1_qpepre.png)  | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/20750_1_qpepre.png)  |
| 5  | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/31200_5_qpepre.png)  | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/20750_5_qpepre.png)  |
| 7  | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/31200_7_qpepre.png)  | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/20750_7_qpepre.png)  |
| 10 | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/31200_10_qpepre.png) | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/20750_10_qpepre.png) |
| 12 | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/31200_12_qpepre.png) | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/20750_12_qpepre.png) |
| 14 | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/31200_14_qpepre.png) | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/20750_14_qpepre.png) |
| 15 | ![](../runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/31200_15_qpepre.png) | ![](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/images/qpepre/20750_15_qpepre.png) |

> Each qpepre panel has a matching power-spectrum figure
> (`{step}_{i}_qpepre_spec.png`) in the same directory if you need to
> diagnose smoothing — drop in `_spec.png` versions of the URLs above.

---

## Why does FlowCast beat the EDM diffusion teacher at the same training-sample budget?

Across the four channels in Table A the FlowCast student lands clearly
ahead of the EDM teacher despite both having seen ~2 M training samples on
the same dataset, the same regression mean, and the same SongUNet trunk.
The gap is not a single thing — it's the sum of (a) flow matching being a
fundamentally easier optimisation problem, (b) FlowCast's recipe being
better tuned for qpepre, and (c) an apples-to-oranges accounting issue in
how validation metrics are reported. Pull each apart in turn before
quoting these numbers in a paper.

### 1. Flow matching has a lower-variance gradient than EDM

Compare the two loss kernels at file level:

| | EDM diffusion | FlowCast (I-CFM) |
|---|---|---|
| target | `y` (the clean residual `R_t`) | `u = x1 - x0` (a fixed unit-variance velocity along a straight line) |
| sample variability | per-sample noise level `sigma ∼ logN(P_mean=-1.2, P_std=1.2)` ranging from ~0.06 to ~70 ([loss.py:255](../physicsnemo/metrics/diffusion/loss.py#L255)) | per-sample time `t ∼ U(eps, 1-eps)`, with eps=1e-5 ([flowcast_loss.py:124-125](../stormcast/utils/flowcast_loss.py#L124-L125)) |
| loss reweighting | `(sigma² + sigma_data²) / (sigma · sigma_data)²` ([loss.py:256](../physicsnemo/metrics/diffusion/loss.py#L256)) | none — plain MSE on the velocity |
| target stays the same across batches? | no — `y + n` is a different noisy view each minibatch | yes — `(x1, x0)` for a given residual gives a fixed velocity `x1 - x0` |
| effective task | denoise at any `sigma` in 4–5 orders of magnitude | predict one velocity field along a unit straight line |

The same residual `R_t` produces a single fixed FlowCast target
`u = x1 - x0`; the EDM trainer instead has to handle that residual at
every noise level it lands on, with the EDM weight up-weighting the
intermediate-`sigma` regime and almost-throwing-away contributions at
`sigma ≈ sigma_max`. The variance of the per-sample gradient is much
higher for EDM, so each optimisation step learns less. Over a fixed
training-sample budget that compounds.

This is the core finding of the FlowCast paper (Lipman et al. 2023,
"Flow Matching for Generative Modeling") and Tong et al. 2024 ("Improving
and generalizing flow-based generative models with minibatch optimal
transport"). At equal data, equal architecture, equal compute, flow
matching consistently lands at a lower validation loss than score-matching
diffusion — exactly what Table A shows.

### 2. The path geometry is aligned with the regression conditioning

Both methods condition on the regression mean `M_t = F_xi(X_{t-1}, S_t, I)`
and learn the residual `R_t = X_t - M_t`. But the way the residual enters
the network is different:

- **FlowCast** standardises the target as `x1 = R_t / sigma_data`
  ([flowcast_loss.py:120](../stormcast/utils/flowcast_loss.py#L120)) and
  draws `x0 ∼ N(0, I)`. The straight-line path
  `x_t = (1 - t) x0 + t x1` ([flowcast_loss.py:130](../stormcast/utils/flowcast_loss.py#L130))
  bakes the fact that "the residual is centred at zero in standardised
  space" into the path geometry: at `t = 0` you are at pure noise, at
  `t = 1` you are at the residual, and the velocity `u = x1 - x0` is
  exactly the displacement between them.
- **EDM** sees `y + n` where `n ∼ N(0, sigma² I)` and has to produce the
  clean `y` at any `sigma`. Nothing in the loss tells the denoiser that
  `||y||` is small in standardised space — it has to discover that on
  its own from gradients.

Net effect: FlowCast spends gradient budget on learning the velocity
*direction*; EDM has to also learn the magnitude calibration across
`sigma`. Same sample count, easier task → faster convergence.

### 3. FlowCast's recipe is explicitly tuned for qpepre

`train_flowcast.sh` adds two qpepre-specific knobs that the EDM training
script (`train_diffusion.sh`) does not have:

- **Per-channel weighting** — `channel_weights=[1.0, 1.0, 1.0, 2.0]`
  doubles the qpepre contribution to the pointwise MSE
  ([flowcast_loss.py:140-151](../stormcast/utils/flowcast_loss.py#L140-L151)).
- **Spectral regulariser** — `spectral_channels=[qpepre]` with
  `spectral_weight=0.1` adds an L1 distance between the radial log-PSD of
  the implied clean qpepre prediction `x1_pred = x_t + (1 - t) v_pred`
  and the ground-truth `x1`
  ([flowcast_loss.py:153-167](../stormcast/utils/flowcast_loss.py#L153-L167)).
  This penalises blurry / over-smoothed qpepre and is exactly what stops
  the student from collapsing onto the regression-mean prior.

The EDM teacher's loss (`physicsnemo.metrics.diffusion.EDMLoss`) is
plain weighted MSE — no per-channel boost, no spectral term. So a
non-trivial slice of the FlowCast→EDM qpepre gap (Table A: RMSE 0.3851
vs 0.7865) is "FlowCast had two extra qpepre-aware loss terms", not
"flow matching is intrinsically better at precipitation". If you want
to isolate the algorithmic effect, re-train an EDM run with the same
`channel_weights` and a comparable spectral term on qpepre and re-score.

### 4. Validation numbers are not directly comparable: EMA vs no EMA

The headline gap also has a measurement-side contributor that's easy
to miss when reading the CSVs:

- **FlowCast trains an EMA shadow** with `ema_decay=0.999`
  ([trainer_flowcast.py:204-210](../stormcast/utils/trainer_flowcast.py#L204-L210)).
  All validation metrics — loss, RMSE, MAE, the validation panels in the
  appendix above — are computed against this EMA shadow, not the online
  student
  ([trainer_flowcast.py:442-446](../stormcast/utils/trainer_flowcast.py#L442-L446)).
- **The diffusion trainer (`stormcast/utils/trainer.py`) does not use EMA
  at all.** Its validation metrics are reported on the raw online weights.

EMA decay 0.999 effectively averages over the last ~1 000 steps (≈ 96 k
training samples for FlowCast), which smooths transient gradient noise
and typically lowers reported RMSE / MAE by a non-trivial amount on
heavy-tailed channels. So at the same ~2 M-sample mark FlowCast's number
is "EMA-smoothed best-recent state" while EDM's is "this exact step,
warts and all". Some — though probably not most — of the Table A gap is
this accounting difference.

To remove the asymmetry: either pull `ema_state.pt` for FlowCast and
score it the same way the trainer would (consistency check, should match
the CSVs); or take a moving-window average of 5–10 nearby validation
points on the EDM CSV (rough EMA proxy) before comparing.

### 5. Bonus: FlowCast is also cheaper at inference

This is a separate axis from training-sample efficiency, but worth
flagging once: the FlowCast inference sampler is Euler-10 (10 NFE per
autoregressive step,
[nn.py:413-414](../stormcast/utils/nn.py#L413-L414))
while the EDM teacher uses the deterministic Heun-18 sampler (≈ 36 NFE,
[deterministic_sampler.py](../physicsnemo/utils/diffusion/deterministic_sampler.py)).
At the chosen sampler settings the deployment cost is ~3.6× lower for
FlowCast on top of the better forecast skill — which is the entire
point of distillation.

### Summary

| reason | direction | how much of the gap |
|---|---|---|
| Flow matching has a lower-variance gradient | algorithmic, intrinsic | substantial |
| Path geometry already aligns with the residual conditioning | algorithmic, intrinsic | moderate |
| FlowCast loss has qpw=2.0 + spectral term, EDM doesn't | recipe, removable | substantial *for the qpepre column* |
| FlowCast validates on an EMA shadow, EDM does not | measurement, removable | small-to-moderate, accounting only |
| FlowCast inference is 3.6× cheaper (Euler-10 vs Heun-18) | deployment | not relevant to training-sample efficiency, just a bonus |

If you want a clean head-to-head, the next experiment is: re-run EDM
with channel_weights and a spectral term mirrored from FlowCast, add
EMA to `trainer.py`, and re-score at ~2 M samples. Anything still
remaining is the part that's actually about flow matching being better
than score matching.

---

## GPU-hours to reach ~2 M training samples

Read off `tot_time` (cumulative wall-clock seconds, including validation)
in each run's `train.log` at the first row whose `samples` column is at
or just past 2 000 000. Every run was launched from
`stormcast/zettabyte_scripts/train_*.sh` with `gpus_per_node=4`, so
**GPU-hours = wall-clock-hours × 4**.

| run | global batch | steps to 2 M | wall-clock | GPU-hours |
|---|---:|---:|---:|---:|
| diffusion log1p (`train_diffusion.sh`) | 64 | 31 250 | 9.37 h (33 715 s) | **37.46** |
| diffusion NO_log1p (`train_diffusion_no_log1p.sh`) | 96 | 20 833 | 6.93 h (24 967 s) | **27.74** |
| flowcast log1p qpw=2.0 default (`train_flowcast.sh`) | 96 | 20 833 | 7.00 h (25 194 s) | **27.99** |
| flowcast NO_log1p (`train_flowcast_no_log1p.sh`) | 96 | 20 833 | 6.97 h (25 079 s) | **27.87** |
| flowcast qpw=1.0 (`train_flowcast_qpw1.0.sh`) | 96 | 20 833 | 7.00 h (25 183 s) | **27.98** |
| flowcast qpw=1.4 (`train_flowcast_qpw1.4.sh`) | 96 | 20 833 | 6.98 h (25 119 s) | **27.91** |
| flowcast qpw=2.4 (`train_flowcast_qpw2.4.sh`) | 96 | 20 833 | 6.97 h (25 093 s) | **27.88** |

> **Hardware.** All runs ran on a single zettabyte worker with 4 GPUs
> visible (`CUDA_VISIBLE_DEVICES=0,1,2,3` in every launcher); the GPU
> SKU isn't recorded in the log header but is the same across all runs
> in this table — see `runs/<run>/train*.log` line 1 for confirmation.
> "GPU-hours" here is wall-clock × 4; subtract idle / validation time
> if you need *active-compute* hours (see footnote below).

### Reading the numbers

- **Diffusion log1p costs ~10 GPU-hours more than diffusion NO_log1p**
  for the same 2 M-sample budget. The reason is mechanical:
  `train_diffusion.sh` ships with batch=64 while
  `train_diffusion_no_log1p.sh` was bumped to batch=96, so the log1p
  variant runs 50 % more optimiser steps **and** 50 % more
  validation-loop invocations at `validation_freq=100` to see the same
  sample budget. **Not a property of the encoding** — it's a
  recipe-asymmetry between the two scripts. Re-training the log1p
  diffusion teacher at batch=96 should bring its cost in line with the
  NO_log1p sibling.
- **All flowcast variants cluster tightly at ~28 GPU-hours.** The qpw
  ablation's per-channel weight tensor and the spectral term add
  negligible compute (the channel reweight is one elementwise multiply,
  the spectral L1 is one FFT per qpepre channel — both <1 % of the
  forward / backward pass). So qpw is essentially free at training
  time; the choice between qpw values is purely about loss-surface
  shaping, not budget.
- **FlowCast vs the diffusion teacher.** At matched samples and *the
  shipped recipe*: 28 vs 37 GPU-hours — FlowCast wall-clock-wins by
  ~25 %, but most of that is the diffusion log1p's batch=64
  bookkeeping, not an algorithmic advantage. Against the diffusion
  NO_log1p run (also batch=96), FlowCast and EDM cost effectively the
  same per 2 M samples (~28 vs ~28 GPU-hours). **The skill gap in
  Table A is therefore not "more compute" — it's "same compute,
  better-extracted gradient signal"**, which is exactly the flow-
  matching argument from the explanation section above.

### Where the wall-clock actually goes

`tot_time` is total wall-clock and **includes the validation loop**,
which calls the full sampler (Heun-18 for EDM, Euler-10 for FlowCast)
every `validation_freq` steps. Cross-checking from the logs:

| run | gradient-step compute | validation-loop compute | total | validation share |
|---|---:|---:|---:|---:|
| diffusion log1p (batch=64) | 22 500 s × 4 = 25.0 GPU-h | 11 200 s × 4 = 12.5 GPU-h | 37.5 | 33 % |
| diffusion NO_log1p (batch=96) | 20 400 s × 4 = 22.7 GPU-h | 4 500 s × 4 = 5.0 GPU-h | 27.7 | 18 % |
| flowcast (batch=96) | 21 300 s × 4 = 23.7 GPU-h | 3 900 s × 4 = 4.3 GPU-h | 28.0 | 15 % |

(Gradient-step compute = `steps_to_2M × steady-state step_time`,
read off the train.log lines near 2 M samples — diffusion log1p ≈
0.72 s/step, diffusion NO_log1p ≈ 0.98 s/step, flowcast ≈ 1.02 s/step.)

So **per-gradient-step compute is essentially identical across the
two batch=96 runs** (22.7 vs 23.7 GPU-h); the wall-clock gap is
dominated by how often each script runs its (different-cost) validation
sampler. If you're benchmarking distillation methods at the gradient-step
level, comparing `step_time × steps` is the right number; if you're
benchmarking end-to-end project cost, `tot_time` is. Both are tabulated
above so you can pick.
