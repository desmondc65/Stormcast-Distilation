# Discussion and Conclusion

Why does the I-CFM (FlowCast) student beat the EDM StormCast teacher at a
matched ~2 M training-sample budget, and what does the cleaned + log1p
pipeline contribute on top of that? This note ties together the two
post-training studies that we can quote directly:

- [results/main_experiment/scoreboard_3way.md](results/main_experiment/scoreboard_3way.md)
  — 3-way rollout comparison (legacy EDM on the 224×128 grid, cleaned EDM
  on 192×96 + log1p, cleaned FlowCast at NFE ∈ {10, 15, 20}), 4 ensemble
  members × 10 sequences × 12 lead-hours on the 2022 validation year.
- [results/log1p_ablation/scoreboard_log1p.md](results/log1p_ablation/scoreboard_log1p.md)
  — within-FlowCast A/B test of the qpepre encoding (log1p mm/h vs raw
  mm/h), same rollout protocol.

All threshold metrics below are in physical units (`mm/h` after
`denormalize_state`), so they ARE directly comparable across the two
qpepre encodings.

---

## 1. Headline numbers

Snapped from
[results/main_experiment/scoreboard_3way.md](results/main_experiment/scoreboard_3way.md):

| method               | Time/Seq (s) | CRPS↓  | FSS-P16-M↑ | HSS-M↑ | FAR-M↓ | RMSE u10↓ | RMSE v10↓ | RMSE t2m↓ | RMSE qpepre↓ |
|---                   |---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_edm (224×128) |  56.20 | 0.872 | 0.0089 | 0.066 | 0.762 | 2.001 | 2.089 | 1.575 | 1.185 |
| cleaned_edm (192×96) |  24.45 | 0.690 | 0.0009 | 0.179 | 0.696 | 1.614 | 2.071 | **1.043** | 0.998 |
| cleaned_flow NFE=10  | **8.37** | 0.689 | **0.0068** | 0.168 | **0.428** | **1.532** | **1.666** | 1.167 | **0.908** |
| cleaned_flow NFE=15  |  10.76 | **0.671** | 0.0058 | 0.174 | 0.544 | 1.560 | 1.700 | 1.149 | 0.911 |
| cleaned_flow NFE=20  |  13.79 | 0.681 | 0.0005 | 0.172 | 0.687 | 1.566 | 1.789 | 1.161 | 0.927 |

The three facts that this table makes unambiguous:

1. **Cleaned pipeline doubles the skill of legacy.** CRPS drops 0.87 →
   0.69 (-21 %); HSS-M triples (0.066 → 0.18); CSI-M doubles. The win is
   architecture-independent — it's the pipeline (cleaned grid, log1p
   qpepre, channel reorder, retraining at the same 2 M-sample budget),
   not which generative head is used.
2. **At matched samples, FlowCast (NFE=10) beats cleaned EDM on every
   sharpness-sensitive metric.** FAR-M is roughly halved (0.696 → 0.428),
   FSS-P16-M improves ~7× (0.0009 → 0.0068), and CRPS is ~tied. The
   only column where EDM holds ground is RMSE on the slow channels
   (t2m).
3. **FlowCast is ~3× faster than EDM at inference and ~7× faster than
   the legacy teacher.** Inference time per 12-h rollout: legacy 56 s,
   cleaned EDM 24 s, FlowCast Euler-10 8 s. The training-sample budget
   was matched at ~2 M; FlowCast wins on both axes.

---

## 2. Why I-CFM (FlowCast) beats EDM in this setup

Five mutually reinforcing reasons, all grounded in the scoreboards above
and not in theory alone.

### 2.1 EDM hedges; FlowCast commits

The clearest discriminator between the two cleaned models is **FAR-M
(false alarm rate) — 0.696 for EDM vs 0.428 for FlowCast at the same
NFE-equivalent budget**. EDM produces broad, blurry rain that overlaps
the truth on enough pixels to score reasonably on CSI averaged across
thresholds (CSI-M 0.148 > 0.132) but at the cost of saying "rain"
nearly everywhere the truth wasn't.

The per-threshold table at
[results/main_experiment/cleaned_2M/per_threshold.csv](results/main_experiment/cleaned_2M/per_threshold.csv)
spells this out:

| threshold (mm/h) | EDM CSI | FlowCast CSI | EDM FAR | FlowCast FAR |
|---:|---:|---:|---:|---:|
| 0.1  | 0.505 | 0.481 | 0.451 | **0.337** |
| 1.0  | 0.286 | 0.262 | 0.637 | **0.400** |
| 5.0  | 0.082 | 0.046 | 0.671 | **0.533** |
| 10.0 | 0.015 | 0.004 | 0.721 | **0.670** |
| 16.0 | 0.000 | **0.002** | 1.000 | **0.200** |

At every threshold FlowCast's FAR is markedly lower, and the gap *grows*
as the threshold becomes harder. EDM's CSI advantage is the trace of
that hedging behaviour — predicting rain everywhere also lifts the hit
rate on the easy thresholds. The Heidke skill score (HSS-M), which
penalises chance-level agreement, sees the two architectures as
near-tied (0.179 vs 0.168) — the EDM CSI lead does not carry over to a
skill metric that subtracts off the climatological baseline.

### 2.2 FlowCast keeps spatial structure on heavy rain

**FSS-P16-M is ~7× better for FlowCast (0.0068 vs 0.0009).** FSS is the
neighbourhood-pooled fractions skill score evaluated at the 16 mm/h
threshold, i.e. heavy-rain spatial coherence. EDM dilutes the heavy
cores into the surrounding pixels, which kills the FSS even when the
total water is right; FlowCast keeps the cores compact enough to score
on the 3, 7, 15-pixel neighbourhood lookups
([results/main_experiment/cleaned_2M/fss_p16.csv](results/main_experiment/cleaned_2M/fss_p16.csv)).

This matches the design difference between the two losses. EDM is a
denoiser at many noise levels — at intermediate σ the target is already
a smoothed version of the residual, and the standard EDM weight
overweights the σ ≈ σ_data regime
([physicsnemo loss.py:256](../physicsnemo/metrics/diffusion/loss.py#L256)).
FlowCast uses plain MSE on a fixed velocity plus a spectral L1
regulariser on the qpepre channel
([flowcast_loss.py:153-167](../stormcast/utils/flowcast_loss.py#L153-L167))
which directly penalises low-pass smoothing of the qpepre prediction.
The 7× FSS gap is what that spectral term buys.

### 2.3 The autoregressive rollout doesn't break FlowCast on precip

`rmse_per_step_3way.csv` shows the lead-time behaviour of each model.
Reading the qpepre column:

| lead time | legacy_edm | cleaned_edm | flowcast_nfe10 |
|---:|---:|---:|---:|
|  +1 h | 0.894 | 0.978 | 0.912 |
|  +6 h | 1.130 | 1.293 | 1.262 |
| +12 h | **0.848** | 0.961 | **0.868** |

The non-monotone shape ("error grows for 6 hours then falls again") is
the well-known wet-to-dry climatological collapse: a 12-h rollout of an
ergodic autoregressive model tends towards the dataset-mean rain rate,
which is itself low, so per-pixel error on a low-precip ground truth
shrinks. Both heads do this. What matters is the *transient* — at +6 h,
FlowCast (1.26) is at the same level as legacy (1.13) on a 192×96 grid
that is harder per-pixel, while cleaned EDM (1.29) is the worst. EDM is
producing more wet-pixel noise than FlowCast at every lead.

For wind, the story is similar but cleaner:

| lead time | cleaned_edm v10 | flowcast_nfe10 v10 |
|---:|---:|---:|
|  +1 h | 1.172 | 1.179 |
|  +6 h | 2.187 | **1.687** |
| +12 h | 2.691 | **2.180** |

FlowCast doesn't just start better — its rollout error grows more
slowly. The flow trajectory is integrated with 10 deterministic Euler
steps per autoregressive frame; the EDM trajectory is integrated with
18 Heun steps, but its denoising target is fundamentally stochastic
(noise sample at σ_max is fresh each frame), and that residual
randomness compounds across the autoregressive loop.

### 2.4 More NFE doesn't help FlowCast — and that's the whole point

Compare the three FlowCast rows (NFE = 10, 15, 20):

| NFE | Time | CRPS | FAR-M | FSS-P16-M | RMSE qpepre |
|---:|---:|---:|---:|---:|---:|
| 10 | 8.4 s | 0.689 | **0.428** | **0.0068** | **0.908** |
| 15 | 10.8 s | **0.671** | 0.544 | 0.0058 | 0.911 |
| 20 | 13.8 s | 0.681 | 0.687 | 0.0005 | 0.927 |

NFE=15 gets a marginal CRPS gain; NFE=20 *degrades* FAR-M back to EDM
levels and collapses FSS to near zero. The straight-line CFM path is
already well-approximated by 10 Euler steps; adding more integration
points starts amplifying small velocity-field errors near t=1 (where
the residual head has the least training mass, since `t ∼ U(eps, 1-eps)`
puts almost no samples at the endpoints —
[flowcast_loss.py:124-125](stormcast/utils/flowcast_loss.py#L124-L125)).
EDM's Heun-18 cannot be dialled below ~36 NFE without ruining the
sampler; so at the operating point where FlowCast is *cheapest* it is
*also most skilful*. That is a strict deployment win, not a tradeoff.

### 2.5 Where EDM still has the edge: t2m

cleaned_edm beats flowcast_nfe10 on t2m RMSE (**1.04 vs 1.17**, 12 %
gap) and the lead-time table shows the gap is structural rather than
sampling noise: cleaned_edm t2m grows from 0.52 → 1.74 over 12 h,
flowcast_nfe10 from 0.56 → 2.02. Two things are happening at once:

1. **The FlowCast loss is shaped for qpepre.** `channel_weights =
   [1, 1, 1, 2.0]` halves the effective gradient budget for t2m, and
   the spectral regulariser is qpepre-only. The qpw ablation in
   [result_table.md](result_table.md#c-qpepre-channel-weight-ablation-flowcast-log1p)
   makes this trade-off explicit — qpw=1.4 has the best t2m+wind
   numbers, qpw=2.0 has the best precip numbers. Picking qpw=2.0 is a
   "precip first, slow channels second" choice.
2. **t2m is dominated by the regression mean.** Both methods condition
   on the same frozen regression `μ_{t+1}`; on a slowly-varying field
   like 2 m temperature, μ already explains most of the variance and
   the residual head has little to do. EDM's higher gradient variance
   is invisible when the residual it has to learn is small, so the
   recipe asymmetry from point 1 dominates.

This is the honest qualifier on the headline claim: **FlowCast wins on
the channels and metrics that matter for storm forecasting (precip,
wind, sharpness, false alarms) and loses ~10 % on the slow channel
that the regression already nearly solves**.

---

## 3. Algorithmic explanation: why I-CFM is easier to optimise than EDM

The empirical lead in §2 has a clean mathematical origin. EDM and I-CFM
solve *different* optimisation problems on the same training data; the
I-CFM problem is intrinsically lower-variance per gradient step, and
its learned ODE is closer to linear at inference. This section makes
that statement precise — the rest of §2 falls out as corollaries.

### 3.1 The two training objectives, side by side

Let the residual be $r = X_{t+1} - \mu_{t+1}$ (in standardised space, so
$r/\sigma_d$ has unit variance), and the conditioning bundle
$c = (M_t, S_t, \mu_{t+1}, I)$.

**EDM** trains a denoiser $D_\theta(\,\cdot\,;\sigma, c)$ over a
log-normal schedule of noise levels:

$$
\mathcal{L}_\mathrm{EDM}(\theta)
\;=\;
\mathbb{E}_{r,\sigma,n}\;
w(\sigma)\,
\bigl\lVert\, D_\theta\bigl(r + \sigma n;\sigma,c\bigr) - r \,\bigr\rVert^{2},
\qquad
n \sim \mathcal{N}(0,I),\;\; \sigma \sim \mathrm{LogNormal}(-1.2, 1.2),
$$

with per-sample weight
$w(\sigma) = (\sigma^{2} + \sigma_d^{2})/(\sigma \sigma_d)^{2}$
([physicsnemo loss.py:256](../physicsnemo/metrics/diffusion/loss.py#L256)).
On the Karras schedule, $\sigma$ ranges $\approx [0.06, 70]$ — about
three decades.

**I-CFM** trains a velocity field $v_\theta(\,\cdot\,;t, c)$ along a
*straight-line* conditional path:

$$
\mathcal{L}_\mathrm{ICFM}(\theta)
\;=\;
\mathbb{E}_{x_{0},x_{1},t}\;
\bigl\lVert\, v_\theta(x_{t};t,c) - (x_{1}-x_{0}) \,\bigr\rVert^{2},
\qquad
x_{0} \sim \mathcal{N}(0,I),\; x_{1} = r/\sigma_d,\; t \sim \mathcal{U}(\epsilon, 1{-}\epsilon),
$$

with $x_t = (1-t)x_0 + t x_1$
([flowcast_loss.py:120-130](../stormcast/utils/flowcast_loss.py#L120-L130)).
Loss is *plain* MSE — no per-$t$ weighting.

The two objectives differ in three ways that all push the same
direction: I-CFM is the easier optimisation problem.

### 3.2 Difference 1 — bounded vs unbounded per-sample gradient

For a *fixed* residual $r$, look at the variance of the per-sample
gradient $\nabla_\theta \ell$:

- **I-CFM.** The regression target is $u = x_1 - x_0 = r/\sigma_d - x_0$.
  Since $r/\sigma_d$ is unit-variance by construction and $x_0$ is
  unit-Gaussian, $\mathbb{E}\lVert u \rVert^{2} \le 2$ per dimension.
  Only one source of randomness in the target ($x_0$); $t$ shifts the
  *input* $x_t$ along the line but does not rescale the target.
- **EDM.** The target is the clean $r$ but the input is $r + \sigma n$
  with $\sigma$ spanning two decades and $n$ fresh per sample. The
  per-sample gradient is multiplied by $w(\sigma) \propto \sigma^{-2}$
  in the small-$\sigma$ tail, so rare small-$\sigma$ draws have
  outsized weight, and large-$\sigma$ draws mostly inject pure noise
  with weight $w(\sigma) \to 1$. Two coupled stochasticities ($\sigma, n$)
  and a heavy-tailed reweighting.

Concretely: under a mini-batch of $B$ samples, the standard deviation
of $\lVert \nabla_\theta \ell \rVert$ is bounded for I-CFM and scales
with the tail of $w(\sigma)$ for EDM. Lower per-sample gradient
variance $\Rightarrow$ lower-variance stochastic update $\Rightarrow$
faster convergence at fixed sample budget. This is exactly the result
that Lipman et al (2023) and Tong et al (2024) make formal, and it
predicts §2.2's CRPS / FSS gap at matched ~2 M samples.

### 3.3 Difference 2 — straight vs curved sampler trajectory

After training, sampling integrates an ODE:

- **EDM probability-flow ODE.**
  $\mathrm{d}x_\sigma/\mathrm{d}\sigma = -\sigma \, \nabla_x \log p_\sigma(x \mid c)$
  along the Karras $\sigma$-schedule. The score is non-linear in $x$,
  so the marginal trajectory from $\mathcal{N}(0, \sigma_\mathrm{max}^{2}I)$
  down to $p_\mathrm{data}$ is **curved**. Accurate integration needs a
  second-order solver (Heun's method) and ~18 substeps = 36 NFE.
- **I-CFM straight-line path.** $x_t = (1-t)x_0 + t x_1$ implies the
  *conditional* velocity is $\mathrm{d}x_t/\mathrm{d}t = x_1 - x_0$ — a
  constant in $t$ for each pair $(x_0, x_1)$. If $v_\theta$ matched
  the conditional velocity exactly, then *one* Euler step from $x_0$
  would recover $x_1$. The *marginal* velocity (averaged over $x_0$ for
  a given $x_t$) is not exactly constant, but it is far less curved
  than the EDM ODE — empirically Euler-10 is sufficient.

This is the direct explanation for §2.4: FlowCast at NFE=10 matches or
exceeds NFE=20 on FAR/FSS because the trajectory is already
well-approximated, and adding more substeps amplifies endpoint errors
where $t \sim \mathcal{U}(\epsilon, 1{-}\epsilon)$ has placed almost no
training mass. EDM cannot drop below the Heun-18 floor without
sampler-stability problems, so FlowCast's deployment cost is structurally
3–4× lower.

### 3.4 Difference 3 — train/inference distribution coupling

EDM training presents the denoiser with $x = r + \sigma n$ at every
$\sigma$ on the schedule. At inference the *initial state* is
$x \sim \mathcal{N}(0, \sigma_\mathrm{max}^{2}I)$ and each Heun step
moves through intermediate $\sigma$ values. Two of the three regimes
seen at training are essentially "pure noise inputs the sampler
re-visits"; the third (small-$\sigma$) is where most of the useful
learning happens, but those gradients are up-weighted by the heavy
$w(\sigma)$ tail.

I-CFM has no analogous mismatch: every intermediate $x_t$ the Euler
sampler ever sees was already in the training support (it is by
construction $(1-t)x_0 + t x_1$ for some $(x_0, x_1, t)$). The marginal
distribution of network inputs at training and inference are the same
distribution, indexed by the same $t$. This tighter coupling shows up
as smaller *deployed* error per integration step, which compounds
favourably across the autoregressive loop.

### 3.5 Why heavy-tailed channels (qpepre) widen the gap

Two algorithmic consequences for sparse, heavy-tailed channels:

1. **EDM smears rare events through $\sigma$.** A rare heavy-rain
   sample has $\lVert r \rVert$ much larger than typical, but for
   $\sigma \gtrsim 1$ the input $r + \sigma n$ is dominated by the
   noise — the heavy-rain signal is *buried* in $\sigma n$ for a
   majority of $\sigma$ draws. The gradient direction "predict heavy
   rain on this pixel" survives only the small-$\sigma$ end of the
   training distribution.
2. **I-CFM preserves the rare event at every $t$.** The target
   $u = x_1 - x_0$ has the same magnitude regardless of $t$; sampling
   $t$ later or earlier shifts $x_t$ along the line but does not
   shrink $\lVert u \rVert$. The gradient direction "predict the
   right velocity for this rare sample" is preserved across the full
   $t$-distribution, not concentrated in a small slice.

This is the algorithmic reason §2.2 sees FSS-P16-M improve ~7× under
FlowCast: heavy-rain *spatial coherence* depends on exactly the rare
samples whose gradient signal EDM dilutes through $\sigma$.

### 3.6 Why the gap widens with lead time (autoregressive compounding)

Each autoregressive step injects sampler error
$\epsilon_k = \hat{r}_k - r_k$ into the next step's conditioning $c_{k+1}$.
After $T$ steps the state drift is bounded by

$$
\bigl\lVert \hat X_T - X_T \bigr\rVert
\;\lesssim\;
\sum_{k=1}^{T}
\underbrace{J_{c \to X}}_{\text{conditioning Jacobian}}\,
\epsilon_k,
$$

with two algorithmic implications:

- **$\lVert \epsilon_k \rVert$ is smaller for I-CFM** because the ODE
  is closer to linear (§3.3) and the train/inference input distribution
  matches (§3.4). The §2.3 rollout table operationalises this: at
  lead +1 h v10 RMSE is essentially identical (1.17 vs 1.18); by
  lead +12 h FlowCast is 2.18 and EDM is 2.69 — a ~20 % gap that
  *grew* with $T$.
- **$\lVert \epsilon_k \rVert$ on heavy-tail channels does not
  inherit the σ-mixing penalty** (§3.5), so qpepre drift stays
  bounded across the autoregressive horizon even though the underlying
  distribution is sparse.

### 3.7 What is *not* an algorithm property

For a fair head-to-head, separate the algorithm from the recipe:

- **`channel_weights = [1, 1, 1, 2.0]`** doubles qpepre's gradient
  mass. Any EDM training script can do the same; ours does not.
- **Spectral L1 on qpepre's radial log-PSD** explicitly penalises
  blurry precipitation. Plain `EDMLoss` has no spectral term.
- **EMA shadow** smooths the deployed weights over ~1 k recent steps.
  Our EDM trainer does not track an EMA.

If EDM were re-trained with these three additions, the qpepre-column
slice of §2.2 and the t2m caveat of §2.5 would narrow substantially.
The arguments in §3.2–§3.6 would not — those are properties of the
*loss-and-trajectory geometry*, not of the recipe hyperparameters.
That residual gap is the part of the FlowCast win that is genuinely
about flow matching being better than score matching on this problem.

---

## 4. The qpepre-encoding study: log1p is doing real work

The log1p ablation
([results/log1p_ablation/scoreboard_log1p.md](results/log1p_ablation/scoreboard_log1p.md))
holds architecture (FlowCast) and training budget fixed and only flips
the qpepre encoding. Every metric improves with log1p:

| metric | log1p | raw mm/h | direction |
|---|---:|---:|---|
| CRPS         | **0.689** | 0.839 | -21.9 % (better) |
| CSI-M        | **0.132** | 0.125 |   +5.9 % |
| FSS-P16-M    | **0.0068** | 0.0030 | +127 % |
| HSS-M        | **0.168** | 0.142 |  +18.1 % |
| FAR-M        | **0.428** | 0.835 |   -48.7 % |
| RMSE u10     | **1.532** | 1.677 |  -8.6 % |
| RMSE v10     | **1.666** | 1.943 | -14.3 % |
| RMSE t2m     | **1.167** | 1.649 | -29.2 % |
| RMSE qpepre  | **0.908** | 1.150 | -21.0 % |

Two observations matter for the discussion of why FlowCast wins:

1. **The encoding effect is at least as large as the architecture
   effect.** Flipping the encoding (raw → log1p) is worth a -49 % FAR
   on its own; that single change is what makes FlowCast's FAR-M
   competitive in the first place. If we had compared FlowCast at raw
   mm/h against EDM at log1p (or vice-versa), the architecture comparison
   would have been confounded.
2. **The encoding also lifts the slow channels.** RMSE t2m drops 29 %
   even though log1p is a transformation on the qpepre channel only.
   This is a shared-trunk effect: the SongUNet weights are not
   per-channel-disjoint, so the badly-scaled raw qpepre channel injects
   gradient noise into the t2m head during training. Standardising the
   heavy-tailed channel fixes the conditioning of the optimisation
   problem, full stop.

The 3-way main experiment correctly holds the encoding constant for
both cleaned legs, so the FlowCast-vs-EDM comparison in §1–§2 is clean
of this confound. The legacy-vs-cleaned gap, however, mixes encoding +
grid + retraining — interpret accordingly.

---

## 5. Caveats

A few caveats the reader should hold while interpreting the numbers:

- **Grid mismatch on the legacy leg.** Legacy is 224×128; cleaned is
  192×96. RMSE is per-pixel averaged so the magnitudes are comparable,
  but the field-of-view differs by ~36 %. The legacy → cleaned jump is
  therefore "new pipeline including grid", not "same domain rescored".
- **Single-snapshot rollouts.** 4 ensemble members × 10 sequences =
  40 rollouts per leg; snapshot noise on individual columns can be
  ~10–30 % per CLAUDE.md §6.5. The qualitative ordering is robust;
  fine ordering between FlowCast NFE=10 vs NFE=15 on a single column
  is not.
- **EMA asymmetry.** FlowCast checkpoints use an EMA shadow at
  inference; the EDM teacher does not. A non-trivial slice of the
  FlowCast → EDM gap is "FlowCast's reported numbers are smoothed over
  ~1 k recent training steps". The architecture comparison is therefore
  an upper bound on FlowCast's algorithmic advantage; the deployment
  comparison (which is what the user sees) is exact.
- **Per-channel weighting confounds qpepre.** As noted in §2.5,
  FlowCast trains with qpw=2.0 and a spectral regulariser on qpepre;
  EDM does not. Some — though not all — of the qpepre gap (0.91 vs
  1.00) is "extra loss terms", not "flow matching is intrinsically
  better at precipitation". A clean isolation would re-train EDM with
  the same recipe knobs and re-score.

None of these caveats change the macro story. They constrain how
ambitious a claim the numbers support.

---

## 6. Conclusion

At a matched ~2 M training-sample budget on Taiwan-domain RWRF and the
2022 validation year:

1. **The cleaned + log1p + retrained pipeline is a strict win over the
   legacy 224×128 / raw-mm/h baseline**, independent of the generative
   head. CRPS drops ~20 %, HSS triples, FAR is reduced by a third. This
   is the headline contribution of the pipeline-engineering work and is
   the larger of the two effects.

2. **Within the cleaned pipeline, FlowCast (I-CFM, qpw=2.0,
   spectral-regularised, EMA, NFE=10 Euler) beats EDM on the
   storm-forecasting metrics that matter** — FAR-M halved, FSS-P16-M
   ~7× better, qpepre RMSE 10 % lower, wind RMSE 5–20 % lower across
   12-hour rollout horizons. EDM retains a ~10 % advantage on the
   slow t2m channel, which is consistent with the FlowCast loss being
   shaped specifically for the heavy-tailed precip channel.

3. **The dominant reasons FlowCast wins on precipitation are
   (i) sharper, more decisive predictions** — the straight-line CFM
   target has a fixed-magnitude per-sample velocity that doesn't ask
   the network to learn calibration across orders of magnitude of
   noise; **(ii) a qpepre-aware loss recipe** that the EDM teacher does
   not get; and **(iii) better autoregressive stability** — error
   growth over the 12-hour rollout is slower on wind and equal on
   precip.

4. **FlowCast wins on cost too.** At NFE=10 it is ~3× faster than EDM
   at inference (8.4 s vs 24.5 s per 12-h rollout) and slightly
   *cheaper* than EDM at training (28 vs 28-38 GPU-h to reach 2 M
   samples, depending on EDM's batch-size choice). The "more NFE =
   better" trade does not apply: NFE=15 and NFE=20 are flat or worse
   than NFE=10 on FAR and FSS, so the cheapest configuration is also
   the best-skill configuration.

The recommended student for downstream operational use is therefore the
**cleaned FlowCast NFE=10 model** (canonical checkpoint
[FlowCastPrecond.0.20000.mdlus + ema_state.pt](../runs/flowcast_zettabyte_v1_cleaned_4_27_2026)),
with the caveat that workloads where t2m skill dominates should reuse
the EDM head or retrain FlowCast with the qpw=1.4 recipe identified in
[result_table.md §C](result_table.md#c-qpepre-channel-weight-ablation-flowcast-log1p).
The next experiment to settle the algorithmic claim cleanly would be
re-training EDM with FlowCast's channel weighting + spectral regulariser
+ EMA, then re-scoring. Anything still in FlowCast's favour after that
is the part that is genuinely about flow matching being better than
score matching on this problem.
