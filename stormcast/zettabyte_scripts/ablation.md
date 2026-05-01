# Ablation plan: FlowCast vs EDM, on the cleaned-with-log1p dataset

This file lays out the runs needed to credibly answer three research questions
in the thesis:

1. **Does FlowCast outperform the EDM teacher** at matched compute (NFE),
   especially on precipitation?
2. **Does the spatial cleaning** (192×96 crop and re-chunking) carry its own
   weight, or does it merely make the U-Net divisible by 32?
3. **Does the log1p qpepre transform** improve precipitation skill, or is it
   neutral now that the qpepre std is already in the ~0.4 regime?

The two launchers under comparison are
[`train_diffusion.sh`](train_diffusion.sh) and
[`train_flowcast.sh`](train_flowcast.sh). They share the dataset, the frozen
regression mean `M_t`, and the channel order; they differ only in the
generative method (EDM-precond + Heun/deterministic sampler vs FlowCast
CFM + Euler) and in optimiser shape (FlowCast adds AdamW + cosine + warmup,
EDM uses Adam + linear warmup).

---

## 1. Run matrix (5 generative + 1 regression)

Each row uses a regression checkpoint that was trained on the **same**
qpepre representation it sees at training time, paired by step number across
its log1p group. Concretely there are two regression runs: R0 (log1p, shared
by D1/F1/F3) and R0_raw (raw mm/h, shared by D2/F2). Reusing R0 across the
raw-mm/h rows would feed F_xi inputs from a different qpepre distribution
than it was trained on AND make M_t live in log1p space while X_t lives in
raw space — i.e. the residual on the qpepre channel would conflate the
log1p ablation with a regression I/O mismatch on the very channel the
ablation is about. Both regression runs share the same
`kept_HighRes_channels=[u10, v10, t2m, qpepre]`; anything not listed below
is held to the launcher defaults so the diff within each pair is exactly
one knob.

| #       | name                              | dataset | qpepre on disk | method     | sampler @ valid | what it tells you                              |
|--------:|-----------------------------------|---------|----------------|------------|-----------------|------------------------------------------------|
| R0      | `regression_cleaned_log1p`        | cleaned | log1p          | regression | n/a             | shared `M_t` for D1 / F1 / F3 (this is `train_regression.sh`) |
| R0_raw  | `regression_cleaned_NO_log1p`     | cleaned | **raw mm/h**   | regression | n/a             | shared `M_t` for D2 / F2 (this is `train_regression_no_log1p.sh`) |
| D1      | `edm_cleaned_log1p`               | cleaned | log1p          | EDM (`σ_data=0.5`) | Heun, 18 steps  | EDM teacher baseline (this is `train_diffusion.sh`) |
| F1      | `flowcast_cleaned_log1p`          | cleaned | log1p          | FlowCast   | Euler, 10 steps | FlowCast student baseline (this is `train_flowcast.sh`) |
| D2      | `edm_cleaned_NO_log1p`            | cleaned | **raw mm/h**   | EDM        | Heun, 18 steps  | log1p ablation (this is `train_diffusion_no_log1p.sh`) |
| F2      | `flowcast_cleaned_NO_log1p`       | cleaned | **raw mm/h**   | FlowCast   | Euler, 10 steps | log1p ablation × method                        |
| F3      | `flowcast_original_log1p_pad`     | original 224×128 | log1p   | FlowCast   | Euler, 10 steps | does the spatial crop matter? (see §3 caveat)  |

D1 vs F1 = the headline FlowCast-vs-EDM comparison. D1 vs D2 (and F1 vs F2)
= the log1p ablation. F1 vs F3 = the spatial-crop ablation.

The minimum subset for a credible thesis chapter is **R0, R0_raw, D1, F1,
D2, F2** (four generative runs + two regression runs — drops F3 if compute
is tight, since the spatial crop is mostly a U-Net-divisibility argument
and the empirical effect should be small).

### Per-row launcher diffs (relative to the current scripts)

* **D2 / F2 — log1p OFF.** Build a sibling cleaned dataset whose `qpepre`
  channel is in raw mm/h (no clean_zarr log1p step, recomputed mean/std on
  the raw values). Concretely: re-run `clean_zarr.py` with the qpepre
  transform commented out, or add a flag, write the result to
  `…_cleaned_4_27_2026_raw/`, then point `dataset.location` there and set
  `dataset.qpepre_log1p=false` so `denormalize_state` is the identity on
  the qpepre channel. **Don't** mix log1p data on disk with
  `qpepre_log1p=false` (or vice versa) — that's silently miscalibrated.
* **F3 — original spatial extent.** Use the source 224×128 zarr
  (`…/zarr_exp3_L_24_H_24_train_2_5_years_full/`), set
  `dataset.HighRes_img_size=[224,128]`, and apply `log1p` either at clean-time
  (re-run clean_zarr on the original-shape source with the transform on) or
  at load-time via a small wrapper. Do not run F3 against raw mm/h data; you'd
  be conflating the spatial-crop and log1p axes.

---

## 2. Metrics (compute on every run, on the same 2022 validation year)

All evaluated **after `denormalize_state`**, i.e. in real physical units (K,
m/s, mm/h). Run via
[`stormcast/inference.py`](../inference.py) /
[`stormcast/inference_flowcast.py`](../inference_flowcast.py) on the cleaned
valid zarr; produce one NetCDF per run and one Pandas table per metric.

| metric                             | unit       | computed on                                  | aggregated over     |
|------------------------------------|------------|----------------------------------------------|---------------------|
| RMSE per channel                   | physical   | every valid hour, every cell                 | space + time        |
| MAE per channel                    | physical   | same                                         | space + time        |
| CSI(qpepre, ≥ τ)                   | dimensionless | binarise both pred/target at τ, count hits | space + time        |
| FSS(qpepre, ≥ τ, scale = N)        | dimensionless | neighbourhood fraction comparison           | by τ ∈ {0.1,1,5,10,20} mm/h, scale ∈ {1,3,9,27} cells |
| bias(qpepre, ≥ τ)                  | dimensionless | (#pred ≥ τ) / (#target ≥ τ)                  | space + time        |
| frequency-of-exceedance            | dimensionless | per-threshold pixel frequency                | by τ                |
| radial log-PSD on qpepre           | dB         | 2-D FFT, radial bin-mean, log10              | per timestep, then mean/std over time |
| rollout RMSE @ {1, 3, 6, 12} h     | physical   | autoregress N steps, compare last frame      | space + time        |
| rollout CSI(qpepre, ≥ τ) @ same    | dimensionless | same                                       | space + time        |

Two measurement protocols are critical:

* **Sample independent valid pairs.** With dt=1h and 8,759 valid pairs,
  draw a fixed list of, e.g., 1,000 evaluation timestamps shared across all
  runs, so paired t-tests / bootstrap on the same hours are valid. Don't
  let each run pick its own eval set.
* **Match NFE.** EDM-Heun with N denoising steps does 2 NFE per step (Heun
  is 2nd-order). FlowCast-Euler is 1 NFE per step. So `EDM Heun = 18 ⇒ 36
  NFE`, `FlowCast Euler = 10 ⇒ 10 NFE`. To compare *at the same compute*,
  add at minimum a `FlowCast Euler @ 36 steps` row (no retraining; just
  override `++inference.flowcast.num_steps=36` at sampling time) and an
  `EDM Heun @ 5 steps = 10 NFE` row. The plot you want is *quality vs NFE*
  for each method, not single-point scores.

---

## 3. The four headline plots / tables

1. **Quality vs NFE curve** — x = NFE on log scale, y = each metric.
   One line for EDM, one for FlowCast. Same regression M_t. Same
   evaluation hours. This is the thesis money plot for question 1.
2. **Per-threshold CSI / FSS bar chart** — six panels (τ ∈
   {0.1, 1, 5, 10, 20} mm/h, scale = 1 + 9 cells). Each panel: a 2×2
   grid (EDM vs FlowCast) × (log1p vs raw). Question 1 + 3 in one image.
3. **Radial log-PSD overlay on qpepre** — separate-line plot for D1, F1,
   D2, F2 + RWRF target. Lets you see whether either method imposes
   spectral bias and whether log1p shifts that bias.
4. **Rollout drift** — RMSE/CSI as a function of rollout step (1, 3, 6,
   12 h), one curve per run. FlowCast often degrades faster on long
   rollouts; this surfaces it.

---

## 4. Statistical protocol

Each metric is computed per-hour (or per-pair of hours for rollout); collect
~1,000 paired observations across runs. Then:

* report **mean ± 95 % bootstrap CI** for each (run, metric) pair;
* paired comparison between F1 and D1 with a **Wilcoxon signed-rank test**
  on the per-hour metric (CSI/FSS bounded; RMSE skewed → non-parametric is
  the safer default);
* don't average over time before testing — that throws away the paired
  structure and inflates p-values.

For ablation honesty, also report the per-run *random-seed* spread. With 4
H100s and ~10–50 h per run, you can afford **2 seeds per row** for the
five-row minimum (10 runs total, ~250–500 H100-hours). If only 1 seed fits,
state it explicitly.

---

## 5. What I expect (and how each result reads)

This is the prediction, useful as a sanity check while runs come in:

| comparison           | expected sign                   | if you see the opposite, suspect…                               |
|----------------------|---------------------------------|------------------------------------------------------------------|
| F1 better than D1 at low NFE (≤10) | yes, by a meaningful margin | FlowCast student wasn't trained long enough, or ema_decay too low |
| F1 vs D1 at high NFE (≥36) | tie or D1 slightly better  | normal; CFM doesn't out-class EDM at high-step inference         |
| D2/F2 vs D1/F1 on RMSE qpepre | log1p worse (raw mm/h has same skill in mean error) | this just confirms RMSE in mm/h vs RMSE in log1p aren't comparable; use CSI/FSS instead |
| D2/F2 vs D1/F1 on CSI ≥ 5 mm/h | **log1p better** (esp. for FlowCast) | a bug in the log1p branch (forgot to apply expm1 before metrics, or stats not recomputed) |
| F3 vs F1 (original vs cleaned domain) | tie within noise | a bug in the original-domain pipeline (wrong padding, wrong stats) |
| Rollout drift, F vs D | F drifts faster after 6 h    | FlowCast is fine; this is a known CFM weakness, not a bug        |

The single most important check: **the log1p qpepre runs (D1, F1) should
beat their raw-mm/h counterparts (D2, F2) on CSI at high precip thresholds
(≥ 5 mm/h)**. If that doesn't show, either the log1p pipeline has a bug
(most likely: forgetting `expm1` somewhere in the metric path) or
heavy-tail tolerance was already absorbed by `channel_weights`.

---

## 6. Practical compute budget (4× H100, fp32)

| run     | steps   | ETA (4× H100) | notes                                              |
|---------|--------:|--------------:|----------------------------------------------------|
| R0      | 16k–30k | ~6–12 h       | already in progress; freeze at the best valid step |
| R0_raw  | 16k–30k | ~6–12 h       | same as R0 but on `_raw` zarr; needed before D2/F2 |
| D1      | 70k     | ~24 h         | EDM teacher; step ~70k matches the paper          |
| F1      | 200–400k | ~50–100 h    | FlowCast paper default; you can stop earlier when valid plateaus |
| D2, F2  | same as D1, F1 | ditto |                                              |
| F3      | same as F1 | ditto    | only if compute allows                             |

Total minimum: **~5 runs × ~50 h ≈ 250 h** of 4×H100 = ~10 days wall-clock
on a dedicated zettabyte instance. Add 50 % for re-runs / hyperparameter
glitches.

---

## 7. Sanity checks to do BEFORE running everything

These are 5-minute checks that prevent multi-day misruns:

1. **Loss curves of D1 and F1 are in different scales** — that's the
   normalised-loss issue you already flagged. Fine, just remember: don't
   compare D1 and F1 by their CSV losses, only by post-`denormalize_state`
   metrics.
2. **`HighRes/stats/{means,stds}.npy` of the no-log1p dataset must be
   recomputed on raw qpepre.** If you reuse the log1p stats with raw data,
   the qpepre std becomes ~9 instead of 0.37 and the channel will dominate
   the loss; you'd see a sane training curve but disastrous wind/temp
   metrics. Verify the std file matches the data on disk.
3. **Channel order across all runs identical**: `[u10, v10, t2m, qpepre]`.
   The launcher comments say so but it's the kind of thing that drifts.
4. **EMA shadow vs raw student**: for both FlowCast and EDM the inference
   weights are *not* the raw `student.parameters()`, they're the EMA
   `ema_state.pt`. Inference scripts already point at the EMA path; verify
   the metric pipeline does too.
5. **Run a 1-hour smoke** of D2 (`total_train_steps=1000`) to confirm the
   no-log1p data loader path actually works before committing to 70k steps.
   Same for F2.

---

## 8. What's *not* in this plan, on purpose

* **Adversarial Diffusion Distillation (ADD).** Out of scope for the FlowCast
  vs EDM comparison; should be its own ablation. The codebase has the
  scaffolding for it (`train_add.py`, `discriminator.py`) but training a
  GAN on sparse precip is its own story.
* **Progressive distillation (PD), consistency distillation (CD).** Same:
  separate ablations against EDM and FlowCast. The CD/PD scripts exist
  and need their own log1p / channel-order updates before they can join
  this matrix on equal footing.
* **Architecture sweeps** (channel widths, attention resolutions). Held
  fixed at the StormCast paper defaults. If FlowCast underperforms,
  *first* check spectral channels / EMA decay / num steps — only then
  reach for architectural changes.
* **More-data ablations** (longer training window, different valid year).
  Nice to have but doubles every run cost. Skip unless a referee asks.
