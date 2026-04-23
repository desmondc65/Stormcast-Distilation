# Metrics, Experiments & Ablations — StormCast Distillation Thesis

**Scope.** Evaluate and compare three student models distilled from the same Taiwan-domain RWRF EDM teacher:

| Method | Shell | Trainer | Inference NFE (default) |
|---|---|---|---|
| Progressive Distillation (PD) | [train_progressive.sh](../train_progressive.sh) | [trainer_progressive.py](../utils/trainer_progressive.py) | 1 (final phase) |
| Consistency Distillation (CD) | [train_consistency.sh](../train_consistency.sh) | [trainer_consistency.py](../utils/trainer_consistency.py) | 1 (EMA weights) |
| FlowCast (CFM) | [train_flowcast.sh](../train_flowcast.sh) | `trainer_flowcast.py` | 10 (Euler) |

**Teacher.** EDM SongUNet `EDMPrecond.0.70000.mdlus`, 18-step Heun sampler, σ_min=0.002, σ_max=80, σ_data=0.5, ρ=7.
**Validation set.** `stormcast_test_valid.zarr` — 2022 full year, 8,760 hourly samples, 4 channels (t2m, u10, v10, qpepre), 224×128 @ 3 km over Taiwan (21.6–25.6°N, 119.75–122.25°E).
**De-normalization stats.** means `[295.98, -1.58, -2.65, 0.182]`, stds `[5.61, 3.84, 5.66, 9.12]`.

All metrics are reported **in physical units** (K, m/s, mm/h) on the **de-normalized** output unless noted.

---

## 1. Metrics Taxonomy

Four metric families, each chosen from a specific paper precedent. Every experiment in §3–§4 is a combination of one or more of these.

### 1.1 Deterministic Pixel-Space (all 4 channels)

| Metric | Definition | Why | Precedent |
|---|---|---|---|
| **RMSE_k** | per-channel root-mean-square error vs RWRF target | Standard regression skill; comparable to StormCast Tab. 1, GenCast Fig. 3 | StormCast §4, GenCast §2.3 |
| **MAE_k** | per-channel mean absolute error | Less sensitive to heavy tail than RMSE; fair for qpepre | FlowCast Tab. 4 |
| **Bias_k** | mean(student − target) | Detects systematic over/under-forecast (esp. qpepre mode collapse) | StormCast Fig. 5 |
| **R²_k** | coefficient of determination | Unit-free skill; 1.0 = teacher, 0 = climatology | GenCast App. C |
| **ACC_k** | anomaly correlation vs RWRF climatology on the validation year | Meteorological standard; detects phase errors even with small amplitude | StormCast §4.1 |

Report per channel × per lead time. Use the 2022 RWRF validation year itself as the climatology reference for ACC (hour-of-day + month mean).

### 1.2 Precipitation-Specific (qpepre only)

Precipitation is sparse and heavy-tailed; pixel RMSE rewards "always drizzle" outputs. Use categorical + spatial skill following **FlowCast Tab. 5** and **StormCast Fig. 7**.

**Thresholds τ (mm/h):** `[0.1, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0]`
- 0.1 = wet/dry binary
- 1.0, 2.5 = light rain
- 5.0, 10.0 = moderate (matches Taiwan climatology 90–95th pct wet-hour)
- 20.0, 50.0 = heavy → extreme (Taiwan typhoon-scale; maps to ~99–99.9th pct)

**Metrics at each τ:**
| Metric | Formula | Source |
|---|---|---|
| **CSI(τ)** | TP / (TP + FP + FN) | FlowCast Tab. 5 |
| **FAR(τ)** | FP / (TP + FP) | FlowCast Tab. 5 |
| **POD(τ)** | TP / (TP + FN) | StormCast §4.2 |
| **HSS(τ)** | Heidke Skill Score | FlowCast Tab. 4 |
| **Bias(τ)** | (TP + FP) / (TP + FN) — frequency bias | StormCast §4.2 |
| **FSS(τ, w)** | Fractions Skill Score at pool window w ∈ {3, 7, 15, 31} px (≈ 9, 21, 45, 93 km) | StormCast Fig. 7, FlowCast §4 |
| **CSI-P(τ, w)** | max-pooled CSI at window w | FlowCast Tab. 5 |
| **CSI-M** | threshold-averaged CSI | FlowCast Tab. 4 |
| **CSI-extreme** | CSI at τ = 20 and τ = 50 mm/h | FlowCast Tab. 5 |

**Distributional checks (qpepre mode-collapse sentinels):**
- **Wet-pixel fraction** `P(X > 0.1)` student vs RWRF
- **Tail quantiles**: Q90, Q95, Q99, Q99.9
- **Q–Q plot** log-scale qpepre student vs teacher samples
- **Wasserstein-1 distance** on intensity histogram (non-zero pixels, log1p transform)

### 1.3 Spectral & Texture

km-scale forecasts are judged on having the right small-scale variance, not just the right mean. Diffusion teachers produce sharp fields; students tend to blur.

| Metric | Definition | Target |
|---|---|---|
| **RAPSD_k(f)** | radially averaged log power-spectral-density (2D FFT → annular bin) per channel | Visual overlay: student ≈ teacher ≈ RWRF down to Nyquist |
| **Log-PSD L1** | ∫\|log P_student(f) − log P_RWRF(f)\| df | Scalar summary; precedent: CD spec §2 + FlowCast App. D |
| **Effective resolution** | smallest wavelength where student log-PSD drops ≥ 3 dB below RWRF | EDM §5, StormCast §4.3 — reveals spectral blur |
| **Gradient-magnitude RMSE** | RMSE of ‖∇X‖ fields | Sharpness of edges (fronts, convective boundaries) |
| **Variogram mismatch** | ∫\|γ_student(h) − γ_RWRF(h)\| dh for h ∈ [3, 90] km | Second-order spatial statistic — complements RAPSD |

### 1.4 Probabilistic / Ensemble (if sampling ≥ 8 members per init)

Stochastic students (CD, CFM) natively support ensembling by re-drawing the latent noise. PD with N=1 step is deterministic; treat it as 1-member.

| Metric | Definition | Precedent |
|---|---|---|
| **CRPS_k** | Continuous Ranked Probability Score, per channel | GenCast §3, FlowCast Tab. 4 |
| **Spread-skill ratio** | ensemble std / ensemble-mean RMSE | GenCast Fig. 2 — target ≈ 1.0 |
| **Rank histogram** | histogram of obs rank in n-member ensemble | GenCast App. C — flatness = calibration |
| **Brier Skill Score (BSS)** | at each precip threshold τ | GenCast §3.2 |
| **REV** | Relative Economic Value at cost/loss ratios {0.05, 0.1, 0.25, 0.5} | GenCast §3.4 (optional) |

Default ensemble size: **n = 10** for the full-year run, **n = 50** for the 30-day case-study window (typhoon events).

### 1.5 Computational Performance

| Metric | How measured |
|---|---|
| **NFE** | forward calls of the student denoiser per forecast |
| **Wall-clock / sample (ms)** | median over 100 samples on 1× A100 (or the NCDR GPU of record), batch 1 |
| **Throughput (samples/s)** | batch 16, sustained over 1000 samples |
| **Peak GPU memory (MiB)** | batch 1 forward pass |
| **Model params (M)** | trainable |
| **Training cost** | total GPU-hours to reach best validation CSI-M |
| **Speedup vs teacher** | teacher_ms / student_ms at matched batch |

---

## 2. Baselines & Reference Points

Every comparison table includes, as rows:

1. **RWRF-climatology** — hour-of-day × month mean on 2022 (trivial lower bound).
2. **Persistence** — X_t = X_{t−1} (harder than climatology at short lead).
3. **Regression-only** — `StormCastUNet.0.14000.mdlus`; deterministic mean, no residual (ceiling for "blurry but unbiased").
4. **EDM teacher @ 18 Heun steps** — quality ceiling. This is the target the students try to match.
5. **EDM teacher @ 4 / 2 / 1 steps** — "naive step-reduction without distillation" — the scalar shows how much distillation actually buys over just sampling less.
6. **Students**: PD (at each phase: 18 → 9 → 4 → 2 → 1), CD (1-step EMA; also 2, 4 multi-step), CFM (1, 2, 4, 10, 25 Euler steps).

Reporting convention: **bold** the best student on each metric, **underline** the teacher, italicize when a student beats the teacher.

---

## 3. Main Experiments

Each experiment is a fixed data protocol; the cells of the result tables are filled by running the baselines + each of {PD, CD, CFM} final checkpoints.

### E1. Full-year single-step deterministic benchmark

- **Data:** every 6th hour of 2022 (1,460 init times → keeps evaluation tractable; documents the sub-sampling).
- **Lead:** one teacher step ahead (1h) given RWRF ground truth `X_{t−1}` and ERA5 `S_t`.
- **Metrics:** §1.1 (all 4 channels) + §1.2 (qpepre) + §1.3 spectral scalar + §1.5 performance.
- **Output:** Table 1 (deterministic skill) + Table 2 (precip categorical) + Figure 1 (RAPSD overlays, one panel per channel).

### E2. Autoregressive rollout (drift evaluation)

- **Data:** 120 initialization times spread over 2022 (every 3rd day, 00Z).
- **Leads:** 1, 3, 6, 12, 24 h (student feeds its own `X̂_{t−1}` back in).
- **Metrics:** RMSE_k, CSI(τ=1, τ=10), wet-fraction, RAPSD at t=24h.
- **Output:** Figure 2 — error-growth curves per method, one subplot per variable + CSI@10mm.
- **Note:** expected finding — PD & CD drift faster than CFM at long lead because CFM's Gaussian-path regularization preserves marginals; document it either way.

### E3. Precipitation case studies

Pick **6 case windows** from 2022 covering:
1. Typhoon Hinnamnor remnants (2022-09-05 → 09-07) — extreme rain
2. Plum-rain front (2022-05-22 → 05-25) — broad moderate rain
3. Afternoon convective burst (2022-07-14 → 07-15, 12–20 UTC) — sharp small-scale
4. Frontal passage, winter (2022-02-11 → 02-13) — cold-front precip
5. Weak-precip day (2022-10-02, < 1 mm total) — false-alarm test
6. Mei-Yu squall line (2022-06-03 → 06-05) — linear MCS

Per case, produce:
- heatmap panels: RWRF / Regression / Teacher-18 / PD-1 / CD-1 / CFM-10 (same colorbar)
- composite radar-like reflectivity rendering of qpepre at lead 1, 3, 6 h
- RAPSD of qpepre for the case window
- CSI(τ=5, 10, 20) time series over the case window

### E4. Ensemble calibration

- Subset of E2: **60 inits**, **10 members each** for CD & CFM; **PD treated as deterministic baseline**.
- Metrics: §1.4 in full. CRPS, spread-skill, rank histograms per channel.
- **Critical for thesis claim** — "distilled student retains probabilistic skill of the teacher".
- Teacher ensemble: 10 members by re-seeding the Heun sampler noise.

### E5. NFE–quality Pareto curves

For each student, sweep NFE and plot the quality/cost frontier.

| Method | NFE sweep |
|---|---|
| PD | phases {18, 9, 4, 2, 1} — same student, different phase checkpoints |
| CD | {1, 2, 4, 8} step multi-step sampling (re-use the 1-step EMA weights, k-step discretization) |
| CFM | {1, 2, 4, 10, 25, 50} Euler steps + {10} RK4 |
| Teacher | {1, 2, 4, 8, 18} Heun steps (no distillation) |

- **X-axis:** NFE (log).
- **Y-axis:** one of {CSI-M, CRPS_qpepre, RMSE_t2m, RAPSD-L1_qpepre}.
- **Output:** Figure 3 — four sub-panels. This is the **headline figure** of the thesis.

### E6. Computational benchmark

Single-table summary of §1.5 for every configuration in E5.

---

## 4. Per-Method Ablations

Keep ablations small and targeted. Each ablation varies **one axis** and holds everything else at the shell-script default.

### 4.1 Progressive Distillation ([train_progressive.sh](../train_progressive.sh))

Default: `initial_num_steps=18`, `target=1`, `steps_per_phase=50000`, `loss_weighting=edm`, `rho=7.0`.

| Ablation | Values | Metric it answers | Precedent |
|---|---|---|---|
| **A1. Loss weighting** | `edm` vs `uniform` vs `truncated_snr` | Does SNR weighting help qpepre at extreme τ? | PD Tab. 1 |
| **A2. Parameterization** | predict-ε vs predict-x vs predict-v | Which target is most stable for qpepre? (PD paper says v) | PD §4 |
| **A3. Phase length** | `steps_per_phase` ∈ {25k, 50k, 100k} | Diminishing returns on training cost | PD App. B |
| **A4. Halving schedule** | 18→1 in {4, 5, 6} phases (default 5) | Is a slower ladder safer? | PD §3 |
| **A5. qpepre transform** | none vs `log1p` vs `asinh` on qpepre pre-loss | Handles sparse heavy tail | CLAUDE.md §7 gotcha |
| **A6. EMA on student** | off vs decay=0.999 | Variance reduction of final weights | EDM §D.3 |

**Promotion curriculum ablation (optional):** compare (a) promote student → teacher at end of each phase vs (b) carry running teacher EMA across phases. PD paper uses (a); (b) is cheaper.

### 4.2 Consistency Distillation ([train_consistency.sh](../train_consistency.sh))

Default: `N_0=2`, `N_total=150`, `huber_c=0.00054`, `ema_decay_init=0.95`, `channel_weights=[1,1,1,2]`, `spectral_weight=0.1`.

| Ablation | Values | Metric it answers | Precedent |
|---|---|---|---|
| **B1. Metric function** | L2 (`huber_c=0`) vs Pseudo-Huber (`0.00054`) vs L1 | CM paper claims LPIPS-like > L2; we use Huber as proxy | CM Tab. 3 |
| **B2. Huber c** | {1e-4, 5.4e-4, 1e-3, 5e-3} | Sensitivity to the c constant — CM-paper is coarse here | CM §5 |
| **B3. Discretization N(k)** | sqrt growth (default) vs linear vs constant=150 | Stability of training | CM §3 |
| **B4. N_total** | {50, 150, 300} | Does a finer grid at end improve fidelity? | CM Tab. 4 |
| **B5. EMA μ_0** | {0.9, 0.95, 0.999} | Effective target-net staleness | CM App. E |
| **B6. Teacher Φ operator** | Euler vs Heun (default Heun) | Gotcha #6 — Heun matches the trained sampler | CLAUDE.md §7 |
| **B7. Channel weights β** | `[1,1,1,1]` vs `[1,1,1,2]` (default) vs `[1,1,1,4]` | Is up-weighting qpepre helping or hurting t2m/u/v? | CD spec §2 |
| **B8. Spectral weight α_spec** | {0, 0.05, 0.1, 0.25, 0.5} | Does radial log-PSD loss close the spectral gap? | CD spec §2 |
| **B9. Spectral channels** | {} vs {qpepre} vs {qpepre, u10, v10} | Spectral loss on wind too? | — |
| **B10. Multi-step inference** | k ∈ {1, 2, 4} | Multi-step CD from CM §3.5 | CM §3 |

### 4.3 FlowCast / CFM ([train_flowcast.sh](../train_flowcast.sh))

Default: `sigma_path=0.01`, `sigma_data=0.5`, `valid_num_steps=10`, Euler solver, `ema_decay=0.999`.

| Ablation | Values | Metric it answers | Precedent |
|---|---|---|---|
| **C1. Path σ** | {0.0, 0.005, 0.01, 0.05} | Probability-path std — tighter = easier matching but lower diversity | FlowCast Tab. 8 |
| **C2. ODE solver** | Euler / Midpoint / RK4 / Heun / Dormand-Prince | FlowCast reports negligible — confirm for Taiwan | FlowCast Tab. 9 |
| **C3. Inference steps** | {1, 2, 4, 10, 25} | Quality-vs-cost frontier; also feeds E5 | FlowCast Fig. 5 |
| **C4. Time-scale** | {1, 1000} (default 1000) | Matches EDM-teacher c_noise scaling; necessary for warm-starting | FlowCast §3.2 |
| **C5. Warm-start from teacher** | random-init vs teacher-init student | Does starting from EDM weights help? | CM §3.4 analogous |
| **C6. Channel weights & spectral loss** | same as B7–B9 | Shared ablation: does the 4-channel reweighting generalize across methods? | — |
| **C7. Conditioning injection** | concat-only (default) vs concat+FiLM | Known to help on multi-channel forecasts | — |

### 4.4 Shared / cross-method ablations

Run on whichever method is the leading candidate at the time the ablation fires.

- **S1. qpepre preprocessing** (`none`, `log1p`, `asinh(x/5)`) — applied in the data loader, not inside the loss.
- **S2. Training-data size** — {4 mo, 12 mo, 29 mo (full)} — learning-curve / data-efficiency plot.
- **S3. Batch size vs LR** — (4, 1e-4), (16, 1e-4), (16, 3e-4), (64, 5e-4) — one-time compute-efficiency sweep.
- **S4. Latent / residual parameterization** — regression residual (default) vs raw X_t prediction — isolates the residual trick's contribution.

---

## 5. Reporting Protocol

### 5.1 Canonical result tables

**Table 1 — Deterministic skill (E1).** Rows: baselines + methods × NFE. Columns: RMSE × {t2m, u10, v10, qpepre}, MAE_qpepre, Bias_qpepre, ACC_t2m. One decimal for RMSE in physical units.

**Table 2 — Precipitation skill (E1).** Columns: CSI(0.1, 1, 5, 10, 20), CSI-M, FSS(w=15 px, τ=5), FAR_mean, wet-fraction ratio, Wasserstein-1.

**Table 3 — Ensemble calibration (E4).** Columns: CRPS per channel, spread-skill ratio per channel, rank-histogram Chi² per channel, BSS(qpepre, τ=10).

**Table 4 — Compute (E6).** Columns: params, NFE, ms/sample, speedup-vs-teacher, peak-mem, training-GPU-hours.

### 5.2 Canonical figures

- **Fig. 1.** RAPSD overlays — 4 panels (one per channel), each showing teacher, RWRF, and 3 students.
- **Fig. 2.** Rollout curves — 4×2 grid (channel × {RMSE, CSI@10} where CSI only for qpepre).
- **Fig. 3.** NFE–quality Pareto — 4 panels (metrics chosen from {CSI-M, CRPS_qpepre, RMSE_t2m, log-PSD-L1_qpepre}).
- **Fig. 4.** Case-study heatmaps — one figure per case in E3.
- **Fig. 5.** Ablation bar charts — one per method, comparing the default against all of 4.x.
- **Fig. 6.** Rank histograms — 4 channels × {CD, CFM, teacher}.

### 5.3 Checkpoint-selection protocol

For each training run:
1. During training, track `valid_loss.csv` and per-channel `rmse_<field>.csv` at `validation_freq`.
2. At the end, **select the checkpoint with best CSI-M(qpepre) on a held-out 5% of validation** (random subset of 2022 hours, fixed seed).
3. Report final metrics on the **remaining 95%** to avoid selection bias.
4. For CD, always load `ema_state.pt` (gotcha in CLAUDE.md §7).
5. For PD, report the final-phase `student_final.mdlus` at each phase N.

### 5.4 Significance testing

- For scalar metrics (RMSE, CRPS, CSI): **stationary block bootstrap** with block length 24 h (daily decorrelation), 1000 resamples, 95% CI.
- For categorical skill: **McNemar's test** on the 2×2 contingency at τ=10 mm/h between the leading student and teacher.
- **Do not report** p-values without the CI — reviewers want effect size.

---

## 6. Execution Order & Priorities

Order matters — cheap results first, so the expensive experiments are informed.

**Phase 1 (first — unblocks everything else).**
1. Lock down E1 pipeline on one already-trained checkpoint per method (even a short pilot run). Verifies that metrics code is correct before expensive training runs.
2. Run baselines: persistence, climatology, regression-only, teacher@{1,2,4,18}.

**Phase 2 (main comparison).**
3. E1 on final PD, CD, CFM checkpoints.
4. E5 — NFE Pareto; this often changes the answer to "which method is best".
5. E2 rollout and E6 compute — cheap extensions of E1/E5.

**Phase 3 (depth).**
6. E3 case studies (6 cases).
7. E4 ensemble calibration (CD, CFM only).

**Phase 4 (ablations).**
8. Run **§4.1 A1, A2** for PD, **§4.2 B1, B2, B7, B8** for CD, **§4.3 C1, C3** for CFM — these are the highest-leverage cells.
9. Fill remaining ablation cells only for the method that leads Phase 2. No need to ablate a losing method exhaustively.

**Phase 5 (cross-method).**
10. §4.4 S1 (qpepre transform) on the leading method. If it wins, re-run Phase 2 with it.

---

## 7. Implementation Notes

- Metric code lives in a new module `stormcast/utils/evaluation/`:
  - `deterministic.py` — §1.1
  - `precip.py` — §1.2 (CSI, FSS, pooled CSI)
  - `spectral.py` — §1.3 (RAPSD, log-PSD-L1, variogram)
  - `probabilistic.py` — §1.4 (CRPS, rank hist, spread-skill)
  - `compute.py` — §1.5 (timing harness)
- Evaluation is **offline**: each method dumps `(init_time, lead, channel) → ndarray` NetCDFs once, then the metric modules read those — this avoids re-running the student per experiment and makes runs reproducible.
- Always **de-normalize** before precipitation metrics (thresholds are in mm/h, not standardized units).
- Random seeds: fix per-init seed for student sampling so tables are reproducible; vary seed only within ensemble members for E4.
- Use the **same 2022 subsampling** (every 6th hour for E1, every 3rd day 00Z for E2) across all methods — the scalar comparison is meaningful only if the data is identical.

---

## 8. What Each Paper Buys Us (precedent map)

| Element | Paper |
|---|---|
| RMSE, MAE, ACC per channel | StormCast §4, GenCast §2.3 |
| FSS at pooling scales {15, 30, 45 km} for precipitation | StormCast Fig. 7 |
| CSI / HSS / FAR at categorical thresholds, CSI-P pooled, CSI-extreme | FlowCast Tab. 5 |
| CRPS, spread-skill, rank histograms, BSS, REV | GenCast §3 |
| Radially averaged log-PSD, effective resolution | EDM §5, FlowCast App. D |
| Training-free NFE sweep on same student | Progressive Distillation Tab. 2, Consistency Models Tab. 4, FlowCast Fig. 5 |
| Loss-weighting ablation (edm / uniform / truncated SNR) | Progressive Distillation Tab. 1 |
| Parameterization ablation (ε / x / v) | Progressive Distillation §4 |
| Huber-c, N(k) schedule, EMA μ₀ ablations | Consistency Models §3, App. E |
| σ_path ablation, ODE-solver ablation | FlowCast Tab. 8, 9 |
| Hinge-GAN discriminator (future ADD method, if added) | Adversarial Diffusion Distillation Tab. 1 |

---

## 9. Open Questions / Decisions to Confirm Before Running

1. **qpepre transform** — confirm whether teacher was trained in raw mm/h or transformed space. Ablation S1 depends on this.
2. **Ensemble size budget** — 10 members × 1460 inits × 3 methods = 43,800 student forecasts for E4. With CFM @ 10 steps on A100 ≈ 2 h wall-clock; verify before committing.
3. **Case-study list** — the 6 cases in E3 should be cross-checked against CWA event logs; the current list is heuristic.
4. **Do we include ADD in the comparison?** The shell script is in progress per CLAUDE.md §3.3; if not ready in time, omit cleanly rather than reporting a half-trained result.
5. **Teacher re-run with Heun vs Euler** — confirm whether the regression + teacher pipeline uses Heun at all leads or mixes solvers; the baseline row `teacher @ 18` must match how the teacher was originally evaluated.
