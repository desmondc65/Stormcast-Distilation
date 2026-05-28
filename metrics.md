# Metrics & Experiment Plan for `runs/`

What to measure with the trained weights in `runs/` (see [CLAUDE.md §6](CLAUDE.md#6-trained-checkpoints-in-runs)).
Metric design is drawn from two sources:

- **[FlowCast](papers/flow_cast.pdf)** — paper that introduced the model
  class we're using; defines CRPS, CSI (and pooled CSI-P16), FSS, HSS, FAR,
  plus the **NFE Pareto** comparison.
- **[StormCast](papers/stormcast.pdf)** — paper that introduced the
  regression + diffusion two-stage residual architecture; defines per-channel
  **RMSE vs. analysis**, **radial power-spectrum density** comparisons, and
  the ensemble **spread-error ratio (SER)** calibration check.

Code we already have to compute most of this:
- [experiment_scripts/_eval_utils.py](experiment_scripts/_eval_utils.py) —
  `MetricAccumulator`, CSI tables, radial PSD, autoregressive rollout.
- [experiment_scripts/compare_diffusion_vs_flowcast.py](experiment_scripts/compare_diffusion_vs_flowcast.py)
  — single-step + ensemble runner: CRPS (kernel), CSI / FAR / POD / HSS,
  FSS-P16, kernel CRPS, panel plots.
- [stormcast/utils/spectrum.py](stormcast/utils/spectrum.py) — `ps1d_plots`
  (radial log-PSD).
- [stormcast/utils/trainer_flowcast.py](stormcast/utils/trainer_flowcast.py)
  — `validation_plot` (per-channel heatmaps).

Dataset facts that constrain every metric in this document:
- **Cleaned grid 192×96** (not 224×128), spacing ≈ 2 km. FSS windows are
  cells, not km — convert before reporting.
- Four output channels: `t2m, u10, v10, qpepre`. Per-channel means
  `μ = [295.98, -1.58, -2.65, 0.182]`, stds
  `σ = [5.61, 3.84, 5.66, 9.12]`.
- **qpepre encoding matters.** Canonical runs use **log1p(mm/h)** then
  per-channel z-score; `_NO_log1p` variants use raw mm/h then z-score. The
  loader's `denormalize_state` undoes the z-score, then `expm1` *only* when
  `qpepre_log1p=True` ([data_loader_rwrf_era5_stable.py:361-377](stormcast/datasets/data_loader_rwrf_era5_stable.py#L361-L377)).
  **Every threshold-based qpepre metric must be computed in mm/h**, not in
  z-score or log1p space.
- **Validation period:** 2022 full year, 8,759 hourly samples
  (`stormcast_test_valid.zarr`). Subsample for fast iteration, but final
  numbers go on the full year.

---

## 0. Notation

Symbols used throughout this document. All numerical quantities are computed
on the **de-normalised** (physical-units) prediction unless otherwise stated.

| Symbol | Meaning |
|---|---|
| `S` | number of validation samples (independent initial times). |
| `T` | number of forecast lead-time steps in a rollout (`T = 1` for single-step). |
| `C` | number of output channels (= 4: `t2m, u10, v10, qpepre`). |
| `H, W` | spatial dimensions (= 192×96 on the cleaned grid). |
| `N` | ensemble size (= 8 for FlowCast in this plan, = 5 in StormCast). |
| `k` | channel index, `k = 1…C`. |
| `s, t, i, j` | sample, lead-time, row, column indices. |
| `y_{s,t,k,i,j}` | ground-truth value (RWRF target). |
| `p_{s,t,k,i,j}` | deterministic prediction (or ensemble mean). |
| `p^{(n)}_{s,t,k,i,j}` | the `n`-th ensemble member's prediction. |
| `Ω = S·T·H·W` | total number of pixel-time-sample triples in an average. |
| `τ` | precipitation threshold (mm/h). |
| `𝟙[·]` | indicator function (0 or 1). |
| `⟨·⟩_A` | mean over the set `A` of indices. |

When an aggregation index range is unambiguous (e.g. "for each channel `k`"),
we drop the subscript and write the mean as `⟨·⟩` or `mean(·)`.

---

## 1. Single-step deterministic skill (per channel)

Source: StormCast §3.2 (per-channel RMSE on HRRR analysis as the truth).

Compute on the de-normalised prediction in **physical units** (K, m/s, mm/h),
for each generative method, against the RWRF target. All sums below run
over `(s, i, j)` for a fixed `k` at single-step (`T = 1`); for rollout
extend the index to `(s, t, i, j)` as well.

### 1.1 RMSE

$$
\mathrm{RMSE}_k \;=\; \sqrt{\frac{1}{|\Omega|}\sum_{s,i,j} \big(p_{s,k,i,j} - y_{s,k,i,j}\big)^2}.
$$

Standard regression skill. Comparable across runs at the **same encoding**;
not comparable across log1p ↔ raw boundary because `expm1` doesn't commute
with `sqrt(mean(·²))`.

### 1.2 MAE

$$
\mathrm{MAE}_k \;=\; \frac{1}{|\Omega|}\sum_{s,i,j} \big|\,p_{s,k,i,j} - y_{s,k,i,j}\,\big|.
$$

Less heavy-tail-sensitive than RMSE. For `qpepre` this is the better scalar
because RMSE is dominated by a small number of extreme-precip pixels.

### 1.3 Bias

$$
\mathrm{Bias}_k \;=\; \frac{1}{|\Omega|}\sum_{s,i,j} \big(p_{s,k,i,j} - y_{s,k,i,j}\big).
$$

Detects systematic over/under-forecast. A persistently positive bias on
`qpepre` is the mode-collapse smoking gun ("always slightly wet").

### 1.4 R² (coefficient of determination)

$$
R^2_k \;=\; 1 \;-\; \frac{\sum_{s,i,j}\big(p_{s,k,i,j} - y_{s,k,i,j}\big)^2}
{\sum_{s,i,j}\big(y_{s,k,i,j} - \bar{y}_k\big)^2},\qquad
\bar{y}_k \;=\; \frac{1}{|\Omega|}\sum_{s,i,j} y_{s,k,i,j}.
$$

Unit-free skill against sample-mean climatology. `R² = 1` is perfect,
`R² = 0` is no better than predicting the long-term mean,
`R² < 0` is *worse* than climatology.

### 1.5 ACC (Anomaly Correlation Coefficient)

Let `c_{k,i,j}` be the per-pixel climatology — the time-mean of the truth
field over the validation period (or a long-term reanalysis climatology if
available):

$$
c_{k,i,j} \;=\; \frac{1}{S}\sum_s y_{s,k,i,j}.
$$

Then the **anomaly fields** are `p' = p − c`, `y' = y − c`, and

$$
\mathrm{ACC}_k \;=\;
\frac{\sum_{s,i,j} p'_{s,k,i,j}\,y'_{s,k,i,j}}
     {\sqrt{\sum_{s,i,j}p'^2_{s,k,i,j}}\;\sqrt{\sum_{s,i,j}y'^2_{s,k,i,j}}}\;\in[-1, 1].
$$

NWP standard. `ACC ≥ 0.6` is the conventional "useful forecast" threshold.
StormCast Fig. 4 reports residual RMSE rather than ACC; ACC is the harder
hurdle for short-range storm-scale forecasts because climatology already
absorbs the seasonal cycle.

### 1.6 Where this lives in code

Implemented in [_eval_utils.py](experiment_scripts/_eval_utils.py)
`MetricAccumulator.update()` for RMSE, MAE, Bias. R² and ACC are not yet
implemented — both are ~5 lines on top of the existing accumulators
`Σ(p−y)²`, `Σ(y−ȳ)²`, `Σ y`, `Σ p'y'`, `Σ p'²`, `Σ y'²`.

### 1.7 What to report

One table with one row per `runs/*` entry, columns `{k × metric}`. Use the
checkpoint-selection rule from
[experiment_scripts/result_table.md](experiment_scripts/result_table.md):
the validation snapshot closest to a fixed training-sample budget (e.g.
2 M, 5 M, 10 M samples), so different batch sizes are comparable.

---

## 2. Single-step precipitation-specific metrics (qpepre only)

Source: FlowCast §A.1.3 — exactly the formulas FlowCast reports.

All thresholds in mm/h on the **de-normalised** qpepre channel.

### 2.1 Threshold-based categorical scores

For each threshold `τ ∈ {0.1, 1, 2.5, 5, 10, 20, 50}` mm/h, binarise both
prediction and truth and tally the contingency table over the *entire*
spatial × temporal × sample volume (FlowCast §A.1.3):

|                     | observed yes              | observed no               |
|---------------------|---------------------------|---------------------------|
| **forecast yes**    | `H` (hits)                | `F` (false alarms)        |
| **forecast no**     | `M` (misses)              | `Z` (correct negatives)   |

Concretely:

$$
\begin{aligned}
H(\tau) &= \sum_{s,i,j}\mathbf{1}[p_{s,i,j} > \tau]\cdot\mathbf{1}[y_{s,i,j} > \tau],\\
F(\tau) &= \sum_{s,i,j}\mathbf{1}[p_{s,i,j} > \tau]\cdot\mathbf{1}[y_{s,i,j} \le \tau],\\
M(\tau) &= \sum_{s,i,j}\mathbf{1}[p_{s,i,j} \le \tau]\cdot\mathbf{1}[y_{s,i,j} > \tau],\\
Z(\tau) &= \sum_{s,i,j}\mathbf{1}[p_{s,i,j} \le \tau]\cdot\mathbf{1}[y_{s,i,j} \le \tau].
\end{aligned}
$$

(FlowCast writes "C" for correct negatives; we use `Z` to avoid clashing
with the channel count.) Then:

#### Critical Success Index (CSI), FlowCast Eq. 3

$$
\mathrm{CSI}(\tau) \;=\; \frac{H(\tau)}{H(\tau) + M(\tau) + F(\tau)}\;\in[0, 1].
$$

Ignores `Z`, so always-dry forecasts can't game it. `CSI = 1` is perfect.

#### False Alarm Ratio (FAR), FlowCast Eq. 2

$$
\mathrm{FAR}(\tau) \;=\; \frac{F(\tau)}{H(\tau) + F(\tau)}\;\in[0, 1].
$$

Fraction of "forecast yes" that didn't happen. Lower is better.

#### Probability Of Detection (POD), aka Hit Rate

$$
\mathrm{POD}(\tau) \;=\; \frac{H(\tau)}{H(\tau) + M(\tau)}\;\in[0, 1].
$$

Recall on the rain class. Trivially gamed by predicting rain everywhere
(POD=1, but FAR≈1 too) — only meaningful alongside FAR.

#### Heidke Skill Score (HSS), FlowCast Eq. 4

$$
\mathrm{HSS}(\tau) \;=\;
\frac{2\big(H Z - M F\big)}
     {(H + M)(M + Z) + (H + F)(F + Z)}\;\in[-1, 1].
$$

Accuracy relative to random chance with the same marginal frequencies.
`HSS = 0` is climatology, `HSS = 1` is perfect, `HSS < 0` is worse than
random.

#### Frequency bias

$$
\mathrm{FBias}(\tau) \;=\; \frac{H(\tau) + F(\tau)}{H(\tau) + M(\tau)}.
$$

`= 1` perfect; `> 1` over-forecasting frequency; `< 1` under-forecasting.
Not in FlowCast but standard in the precipitation literature.

#### Aggregated ("-M" suffix in FlowCast)

The unweighted threshold mean,
$\mathrm{CSI\text{-}M} = \tfrac{1}{|\mathcal T|}\sum_{\tau \in \mathcal T}\mathrm{CSI}(\tau)$, similarly for HSS-M, FAR-M, FBias-M, POD-M.
Report **both** the per-threshold table and the `-M` aggregate.

#### Extreme-event subset (FlowCast Table 5 analogue)

Report `CSI / HSS / FAR` separately at the highest thresholds (e.g. `τ ∈
{20, 50}` mm/h). That's where the mode-collapse failure mode shows up —
the `-M` aggregate hides it because low thresholds dominate the mean.

Existing code: [compare_diffusion_vs_flowcast.py:155](experiment_scripts/compare_diffusion_vs_flowcast.py#L155)
`categorical_scores()`. Already returns CSI / FAR / POD / HSS per threshold.

### 2.2 Pooled CSI (CSI-P16) and FSS

Spatial-tolerance versions of CSI / threshold scores that reward "rain in
the right neighborhood" rather than "rain in the right pixel" — they avoid
the double-penalty problem RMSE has on sparse fields where a slight spatial
offset doubles the error.

#### CSI-P16

Let `MaxPool_w(·)` be non-overlapping max-pooling over `w × w` blocks
(FlowCast uses `w = 16`):

$$
\tilde p \;=\; \mathrm{MaxPool}_w(p),\qquad
\tilde y \;=\; \mathrm{MaxPool}_w(y).
$$

Compute the contingency table `H̃, F̃, M̃, Z̃` on the pooled fields and define

$$
\mathrm{CSI\text{-}P}w(\tau) \;=\; \frac{\tilde H(\tau)}{\tilde H(\tau) + \tilde M(\tau) + \tilde F(\tau)}.
$$

A "hit" only requires *any* pixel inside a `w × w` block to exceed `τ` in
both prediction and truth.

#### FSS (Fractions Skill Score), FlowCast Eq. 5

For an `n × n` neighborhood centred at `(i, j)`, compute the fraction of
pixels exceeding `τ`:

$$
S_p(i,j;\tau,n) \;=\; \frac{1}{n^2}\sum_{(a,b)\in B_n(i,j)}\mathbf{1}[p_{a,b} > \tau],\qquad
S_y(i,j;\tau,n) \;=\; \frac{1}{n^2}\sum_{(a,b)\in B_n(i,j)}\mathbf{1}[y_{a,b} > \tau],
$$

where `B_n(i,j)` is the `n × n` box centred at `(i, j)`. Then

$$
\mathrm{FSS}(\tau, n) \;=\; 1 \;-\;
\frac{\sum_{i,j}\big(S_p(i,j;\tau,n) - S_y(i,j;\tau,n)\big)^2}
     {\sum_{i,j} S_p(i,j;\tau,n)^2 + \sum_{i,j} S_y(i,j;\tau,n)^2}\;\in[0, 1].
$$

`FSS = 1` is perfect spatial agreement at scale `n`; `FSS = 0` means the
forecast neighborhoods have *no* overlap with the observed neighborhoods.
A useful reference: the "no-skill" baseline `FSS_uniform = 2 f / (1 + f²)`
where `f` is the climatological rain frequency at threshold `τ`. Anything
above that is genuine spatial skill.

#### Pooling windows for FSS

FlowCast uses 16-pixel SEVIR blocks (~16 km). Our 2 km cleaned grid →
match the FlowCast spirit by using `n ∈ {3, 7, 15, 31}` pixels =
`{6, 14, 30, 62}` km. The `n = 15` window is the closest analogue to
FlowCast's CSI-P16; `n = 31` is needed because typhoon-scale precip features
in Taiwan are bigger than SEVIR's continental US convection.

Existing code: [compare_diffusion_vs_flowcast.py:193](experiment_scripts/compare_diffusion_vs_flowcast.py#L193)
`fss()` and `_box_pool()`. Already used; just align the window list to
the grid scale.

---

## 3. Probabilistic / ensemble metrics

Source: FlowCast §4.1.2 + StormCast Appendix D. **Requires running each
generative method multiple times with different noise seeds** to get an
ensemble of size `N` (FlowCast: 8; StormCast: 5).

For each pixel `(s, k, i, j)` denote the ensemble
$\{\hat p^{(1)}, \hat p^{(2)}, \ldots, \hat p^{(N)}\}$
and its empirical CDF

$$
\hat F_{s,k,i,j}(z) \;=\; \frac{1}{N}\sum_{n=1}^{N}\mathbf{1}\!\left[\hat p^{(n)}_{s,k,i,j} \le z\right].
$$

### 3.1 CRPS (Continuous Ranked Probability Score)

For a single pixel with truth `y` and forecast CDF `F`:

$$
\mathrm{CRPS}(F, y) \;=\; \int_{-\infty}^{\infty}\!\big(F(z) - \mathbf{1}[z \ge y]\big)^2\,\mathrm{d}z.
$$

Lower is better; for a perfect deterministic forecast (`F` = Heaviside at
`y`) it equals 0; for `N = 1` it reduces to the absolute error `|p − y|`.
The full metric averages this scalar over all `(s, k, i, j)` and reports
per-channel.

Two closed-form estimators are useful in practice:

#### (a) Kernel / empirical CRPS — preferred

Hersbach (2000): for an empirical ensemble of size `N` with sorted members
$\hat p^{(1)} \le \cdots \le \hat p^{(N)}$,

$$
\mathrm{CRPS}_{\mathrm{kernel}}\!\left(\{\hat p^{(n)}\}, y\right) \;=\;
\frac{1}{N}\sum_{n=1}^{N}\big|\hat p^{(n)} - y\big| \;-\;
\frac{1}{2N^2}\sum_{n=1}^{N}\sum_{m=1}^{N}\big|\hat p^{(n)} - \hat p^{(m)}\big|.
$$

No distributional assumption — handles heavy-tailed `qpepre` correctly.
Already implemented in
[compare_diffusion_vs_flowcast.py:204](experiment_scripts/compare_diffusion_vs_flowcast.py#L204)
`crps_field`.

#### (b) Gaussian approximation — FlowCast Eq. 1

Approximates the ensemble as `F ≈ 𝒩(μ̂, σ̂²)` with
$\hat\mu = \tfrac{1}{N}\sum_n \hat p^{(n)}$,
$\hat\sigma^2 = \tfrac{1}{N-1}\sum_n (\hat p^{(n)} - \hat\mu)^2$:

$$
\mathrm{CRPS}_{\mathcal N}(\hat\mu, \hat\sigma, y) \;=\;
\hat\sigma\left(
\frac{y - \hat\mu}{\hat\sigma}\!\left[2\,\Phi\!\left(\tfrac{y - \hat\mu}{\hat\sigma}\right) - 1\right]
+ 2\,\phi\!\left(\tfrac{y - \hat\mu}{\hat\sigma}\right)
- \frac{1}{\sqrt{\pi}}
\right),
$$

where `Φ` and `φ` are the standard-normal CDF / PDF. Cheaper, but biased
on heavy-tailed channels (qpepre). Use **only** if you want to reproduce
FlowCast's reported scalars exactly; otherwise use (a).

#### Normalisation

FlowCast normalises by the dataset's maximum pixel value (`1/255` for
SEVIR's 8-bit VIL, `1/57` for ARSO's dBZ) to make CRPS dimensionless and
comparable across datasets. For us the analogous normaliser is
**`qpepre_max_train` in mm/h** (~200 mm/h in this dataset). Report both
raw mm/h CRPS and the normalised scalar.

### 3.2 Spread-error ratio (SER)

StormCast Eq. 6. Ensemble mean
$\bar p_{s,k,i,j} = \tfrac{1}{N}\sum_n \hat p^{(n)}_{s,k,i,j}$;
ensemble spread (pixel-wise standard deviation across members):

$$
\mathrm{spread}_{s,k,i,j} \;=\; \sqrt{\frac{1}{N-1}\sum_{n=1}^{N}\big(\hat p^{(n)}_{s,k,i,j} - \bar p_{s,k,i,j}\big)^2}.
$$

The ensemble-mean RMSE against truth (per lead time `t`):

$$
\mathrm{RMSE}_{\bar p}(t,k) \;=\; \sqrt{\frac{1}{S\cdot H\cdot W}\sum_{s,i,j}\big(\bar p_{s,t,k,i,j} - y_{s,t,k,i,j}\big)^2}.
$$

Then

$$
\mathrm{SER}(t, k) \;=\;
\frac{\big\langle\mathrm{spread}_{s,t,k,i,j}\big\rangle_{s,i,j}}{\mathrm{RMSE}_{\bar p}(t,k)}
\cdot\sqrt{\frac{N+1}{N}}.
$$

The $\sqrt{(N+1)/N}$ factor is the small-ensemble bias correction.

- **`SER = 1`** → perfectly calibrated.
- **`SER < 1`** → underdispersive (ensemble too tight; truth often outside
  the ensemble).
- **`SER > 1`** → overdispersive.

Plot **per channel** as a function of **lead time** for rollout. StormCast
itself comes out underdispersive (App. D), so a flat `SER ≈ 1` curve on
FlowCast would be a notable result.

Not currently in the codebase — needs to be added to `MetricAccumulator`
or as a post-process on the ensemble tensor returned by `run_method_ensemble`.

### 3.3 Rank histogram (Talagrand diagram)

For each `(s, k, i, j)`, sort the `N + 1` values
$\{\hat p^{(1)}, \ldots, \hat p^{(N)}, y\}$ and record the **rank of the
truth** `y` within them, $r_{s,k,i,j} \in \{1, 2, \ldots, N+1\}$:

$$
r_{s,k,i,j} \;=\; 1 \;+\; \sum_{n=1}^{N}\mathbf{1}\!\left[\hat p^{(n)}_{s,k,i,j} < y_{s,k,i,j}\right].
$$

Tied values are broken with a uniform random index among the ties
(standard convention; otherwise the histogram has spurious spikes on
integer-valued or always-zero pixels — which `qpepre` very much is).

Aggregate over all `(s, i, j)` per channel and per lead time → a histogram
on `{1, …, N+1}` with `N + 1 = 9` bins for `N = 8`.

Interpretation:

- **Flat** → calibrated (every rank equally likely).
- **U-shape** → underdispersive (truth often outside the ensemble; this is
  StormCast's failure mode for qpepre).
- **Bell** → overdispersive (truth always near the centre; ensemble too wide).
- **Slope** → biased ensemble mean.

Pair this with SER — they diagnose the *same* failure but the rank histogram
shows *which tail* is failing. ~30 lines on top of the ensemble tensor.

### 3.4 Probability-Matched Mean (PMM)

StormCast §2.3, §3.2. The plain ensemble mean is smooth, which destroys
extreme intensities — bad for precipitation. PMM keeps the *spatial pattern*
of the ensemble mean but *re-injects* a sharp pixel-value distribution from
a single member.

Formally, for a fixed `(s, k)`, let `r_{i,j} ∈ {1, ..., HW}` be the
ascending **rank** of the ensemble mean `p̄` over all `HW` pixels, and let
$\hat p^{(*)}_{(1)} \le \hat p^{(*)}_{(2)} \le \cdots \le \hat p^{(*)}_{(HW)}$
be the sorted pixel values from a single arbitrary member (or, more robustly,
the pooled sort of *all* `N · HW` member pixels). Then

$$
\mathrm{PMM}_{i,j} \;=\; \hat p^{(*)}_{(\,r_{i,j}\,)}.
$$

Equivalently, in pseudocode:

```python
mean_field = ensemble.mean(axis=0)                  # (H, W) — smooth
ranks      = mean_field.flatten().argsort().argsort().reshape(H, W)
sorted_vals = np.sort(ensemble.flatten())           # length N*H*W, pooled
# downsample to length H*W:
sorted_vals = sorted_vals[::N]                       # take every N-th, ascending
pmm_field  = sorted_vals[ranks]                      # (H, W) — sharp
```

The PMM field has the ensemble mean's spatial structure and any single
member's intensity histogram. Use **PMM-of-ensemble** as the deterministic
forecast for the §2 categorical metrics (CSI, FSS) and compare side-by-side
with using **plain ensemble mean**. StormCast shows PMM is consistently
more skilful than the plain ensemble mean at high `τ` because it preserves
the heavy tail.

---

## 4. Spectral metrics

Source: StormCast §3.3 and Fig. 5 (relative-error PSD).

### 4.1 2-D power spectrum

For a single field `X ∈ ℝ^{H×W}`, the 2-D discrete Fourier transform is

$$
\hat X(k_y, k_x) \;=\; \sum_{i=0}^{H-1}\sum_{j=0}^{W-1} X_{i,j}\,
\exp\!\left[-2\pi\mathrm{i}\left(\tfrac{k_y i}{H} + \tfrac{k_x j}{W}\right)\right].
$$

The 2-D power spectrum is $P(k_y, k_x) = |\hat X(k_y, k_x)|^2$.

### 4.2 Radial average

Define the radial wavenumber index $\|k\| = \sqrt{k_y^2 + k_x^2}$ and bin
into integer radial shells

$$
\mathcal K_\ell \;=\; \{(k_y, k_x)\,:\,\lfloor\|k\|\rfloor = \ell\},\qquad \ell = 0, 1, \ldots, \ell_{\max}.
$$

The **radial power spectrum** is

$$
\bar P(\ell) \;=\; \frac{1}{|\mathcal K_\ell|}\sum_{(k_y, k_x)\in\mathcal K_\ell} P(k_y, k_x),
$$

and we report it in log space, $\log \bar P(\ell)$, averaged over all `S`
validation samples (and `T` lead times for rollout). This is exactly what
[stormcast/utils/spectrum.py](stormcast/utils/spectrum.py) `ps1d_plots`
produces.

### 4.3 Two plots to report

1. **Per-channel log-PSD overlay** (StormCast Fig. 5 left column). Plot
   $\log \bar P^{\mathrm{pred}}(\ell)$ vs $\log \bar P^{\mathrm{truth}}(\ell)$
   for each method on the same axes. Target: pred and truth curves visually
   indistinguishable up to the Nyquist scale.

2. **Relative PSD error** (StormCast Fig. 5 right column):

   $$
   \Delta(\ell) \;=\; \frac{\bar P^{\mathrm{pred}}(\ell)}{\bar P^{\mathrm{truth}}(\ell)} - 1.
   $$

   StormCast's empirical target is $|\Delta(\ell)| < 0.2$ across the
   resolved range. Negative `Δ` at high `ℓ` is the over-smoothing
   signature of regression / under-trained diffusion; positive `Δ` at
   high `ℓ` is the noise-injection failure mode of mis-tuned diffusion.

### 4.4 Where this lives in code

`ps1d_plots` already saves `ps1d_{field}.csv` during training. What's
missing:

- The *relative-error* overlay $\Delta(\ell)$ (truth-normalised, log-x).
- A cross-method comparison plot (one line per method on the same axes).

Both are ~20 lines on the CSVs that already exist.

---

## 5. Autoregressive rollout

Source: StormCast §3.2 (Fig. 4) — RMSE vs. lead time, plus FlowCast's
12-step rollout convention.

The single-step metrics above (§1-§3) measure `t → t+1`. The autoregressive
question is: how do errors grow when the model's own output becomes the
next conditioning input?

Concretely, given the deterministic generative residual sampler `G`
(EDM-Heun, FlowCast-Euler, or just `0` for regression-only), define the
rollout by

$$
\begin{aligned}
M_{t+1} &= F_\xi(X_t, S_{t+1}, I), \\
R_{t+1} &= G\!\left(M_{t+1}, X_t, S_{t+1}, I\right), \\
X_{t+1} &= M_{t+1} + R_{t+1},
\end{aligned}
\qquad t = 0, 1, \ldots, T-1,
$$

with `X_0` the ground-truth initial state at the chosen initialisation
time, `S_{t+1}` the LowRes conditioning at the verification time, and
`I` the invariants. `F_ξ` is the frozen regression network; `G` is the
generative residual sampler. For ensembles, `G` draws a fresh noise seed
per member per step.

| Lead time | What it answers |
|---|---|
| 1 h (= single-step) | Pure generative skill. |
| 3 h | Short-range forecast. |
| 6 h | Medium-range; where StormCast PMM starts losing to HRRR (Fig. 3). |
| 12 h | Where `qpepre` mode collapse usually shows. |
| 24 h | Stability check — does the rollout blow up? |

Recompute **every metric in §1-§3** at each lead time. The minimum useful
plot is RMSE/CSI vs. lead time per channel per method.

Existing code: [_eval_utils.py:402](experiment_scripts/_eval_utils.py#L402)
`autoregressive_rollout` and
[compare_diffusion_vs_flowcast.py:213](experiment_scripts/compare_diffusion_vs_flowcast.py#L213)
`rollout`. Already feeds output back as the next state.

**Initial-condition sampling.** FlowCast uses random initialisation times;
StormCast initialises on 14 representative dates. For us, the **14
typhoon-active days** of 2022 are the natural choice — pick from the dates
with maximum daily mean `qpepre`. Otherwise an unstratified random sample
of 50-100 initialisations will average over too many dry days.

---

## 6. NFE Pareto sweep (FlowCast Fig. 5)

For FlowCast and the EDM teacher, plot **quality vs. number of function
evaluations** (NFE). For each method:

- EDM teacher (Heun sampler): one Heun step ⇒ 2 NFE.
- FlowCast (Euler sampler): one Euler step ⇒ 1 NFE.

Sweep:

| Method | Sampler-step values to test |
|---|---|
| EDM teacher (Heun) | 1, 2, 4, 8, 18, 32 |
| FlowCast (Euler) | 1, 2, 4, 8, 10, 16, 32 |

Quality metric on the y-axis: pick **one** scalar that condenses what you
care about. Two reasonable choices:

- **CRPS-qpepre** (FlowCast's choice; §3.1) — emphasises probabilistic
  precipitation skill.
- **CSI-M-qpepre** at `τ ∈ {1, 5, 10}` mm/h (§2.1) — emphasises
  categorical extreme skill.

FlowCast paper claim to verify: CFM saturates at ~3-10 NFE while diffusion
needs ~20-50. With our checkpoints we can test this directly on the Taiwan
domain — if it doesn't replicate, that's a publishable finding.

Existing code: `flowcast_model_forward(..., num_steps=N)` and
`diffusion_model_forward(..., sampler_args=dict(num_steps=N))` in
[stormcast/utils/nn.py:141,163](stormcast/utils/nn.py#L141). Add a
`--nfe-sweep` loop on top of `compare_diffusion_vs_flowcast.py`.

---

## 7. Experiment matrix mapped to `runs/`

The runs decompose into three orthogonal axes:

| Axis | Levels | Where |
|---|---|---|
| **Encoding** | log1p / raw mm/h | §6.1 vs §6.2 of CLAUDE.md |
| **Generative method** | regression / EDM / FlowCast / CorrDiff | §6.1, §6.4 of CLAUDE.md |
| **qpw (FlowCast only)** | 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4 | §6.3 of CLAUDE.md |

Concrete experiments to run:

### E1 — Architecture comparison at fixed encoding
*Question: does FlowCast beat the EDM teacher on the cleaned log1p dataset?*

Holding encoding = log1p, compare regression / EDM / FlowCast (qpw=2.0).
Report §1 (deterministic, all 4 channels), §2 (qpepre categorical + FSS),
§3.1 (CRPS), §4 (PSD). Use the latest checkpoint of each.

### E2 — Encoding ablation
*Question: does log1p help, and is the answer architecture-dependent?*

Compare each architecture's log1p vs `_NO_log1p` variant. Report §1 for
`{t2m, u10, v10}` (these are scalar-comparable across encodings) and §2
(categorical metrics on mm/h, which neutralises the encoding choice — both
rows decode to mm/h before binarising).

**Trap to avoid.** The standardised log1p RMSE for qpepre is **not**
comparable to the standardised raw-mm/h RMSE because `expm1` doesn't
commute with `sqrt(mean(·²))`. Report mm/h-space numbers only. See
[experiment_scripts/result_table.md](experiment_scripts/result_table.md)
"† footnote" for the worked example.

### E3 — qpw ablation
*Question: what's the optimal qpepre channel weight for FlowCast?*

The seven `flowcast_qpw_ablation/qpw*/` runs already form this matrix.
Add §2.1 categorical scores (CSI / HSS at 1, 10, 50 mm/h) and §3.1 kernel
CRPS-qpepre to the existing `result_table.md`. The §6.5 table in CLAUDE.md
only shows standardised RMSE — categorical scores might re-order the qpw
ranking (mode collapse hurts CSI but not always RMSE).

### E4 — NFE Pareto
*Question: does the FlowCast paper's "saturates at 3-10 NFE" hold on RWRF?*

See §6. One run per `(method, NFE)` cell on a 200-300 sample validation
subset. The full year isn't necessary for an NFE curve — the relative
ordering across NFEs is stable.

### E5 — Rollout
*Question: at what lead time does each method's skill collapse?*

For E1's three architectures, do a 12-hour autoregressive rollout on
14-50 initialisations (typhoon-biased). Report §1 + §2 + §3.1 + §3.2 (SER)
vs. lead time.

### E6 — Ensemble calibration
*Question: are FlowCast ensembles calibrated, or under-dispersive like
StormCast?*

For FlowCast (qpw=2.0 canonical), run §3.1 (CRPS), §3.2 (SER vs. lead time),
§3.3 (rank histogram), §3.4 (PMM) on `N = 8` ensembles. This is the closest
analogue to FlowCast Tab. 6 + StormCast App. D combined.

### E7 — CorrDiff baseline
*Question: do our two-stage residual models beat the
specialised-precipitation CorrDiff baseline?*

Add `corrdiff_diffusion_zettabyte_v1_cleaned_4_27_2026` and
`corrdiff_regression_zettabyte_v1_cleaned_4_27_2026` to E1's table. Use §1
+ §2 on the de-normalised output. Be careful with the step-counter units
(samples seen, not optimizer steps — see CLAUDE.md §6.4).

---

## 8. Reporting checklist

For each experiment above, save:

1. **A CSV per metric × method × variable** — one column per metric, one
   row per (run, lead time). This is the substrate for any plot.
2. **A summary table in `experiment_scripts/result_table.md`** — already
   established convention.
3. **One headline plot per experiment:**
   - E1: bar chart per channel × method (4 channels, 3-4 methods).
   - E2: 2×2 grid (encoding × architecture) on mm/h CSI-M-qpepre.
   - E3: line plot, qpw on x-axis, one curve per metric.
   - E4: NFE Pareto, log-x.
   - E5: 4×4 grid (channel × metric) of curves vs. lead time.
   - E6: rank histogram per channel + SER vs. lead time.
   - E7: same as E1 with two extra rows.
4. **One qualitative panel** per experiment — pick the same 3-4 dates
   (one typhoon, one frontal rain, one dry case) and overlay
   `{truth, regression, EDM, FlowCast, [CorrDiff]}` heatmaps for qpepre.
   The reader does ~80 % of the comparison from this panel.

---

## 9. Quick-start commands

Single-step batch comparison (E1):
```bash
cd experiment_scripts
bash run_compare_diffusion_vs_flowcast.sh
```

NFE Pareto (E4, needs to be added):
```bash
for NFE in 1 2 4 8 10 16 32; do
  python compare_diffusion_vs_flowcast.py \
      --flow-nfe $NFE --diff-nfe $NFE \
      --n-samples 200 --ensemble 4 \
      --out-dir results/nfe_$NFE
done
```

Rollout (E5, hook into `_eval_utils.autoregressive_rollout`):
```python
from experiment_scripts._eval_utils import autoregressive_rollout
# loads checkpoints, runs 12-step rollout on N init dates,
# returns (method, lead, channel, metric) tensor
```

---

## 10. Things explicitly NOT in this plan (and why)

- **DDIM / DPM-Solver comparison.** FlowCast paper uses DDIM as the
  diffusion baseline because that's what their backbone was trained for.
  The EDM teacher in this repo uses Heun, which is the appropriate match —
  adding DDIM would muddy the comparison.
- **Composite reflectivity.** StormCast's headline FSS plots are on radar
  reflectivity (dBZ). We don't have that channel; our precipitation proxy
  is qpepre (mm/h). The qpepre thresholds in §2.1 are the equivalent
  light / moderate / heavy bands.
- **HRRR / RWRF as a baseline.** StormCast and FlowCast both compare
  against the physical model their training set came from. We could do
  this with RWRF re-forecasts if available — flagged but not included in
  the matrix until a re-forecast dataset shows up.
- **Per-region scoring.** Domain is small (Taiwan, 192×96 cells);
  per-region scoring is only meaningful if you stratify by terrain
  (orographic vs. coastal). Worth considering as a follow-up but not a
  default.
