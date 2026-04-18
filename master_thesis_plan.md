# Master's Thesis Plan — StormCast Distillation (Taiwan RWRF)

**Title:** *Few-Step Distillation of a Regional StormCast Diffusion Model for Taiwan Convective-Scale Weather Forecasting*

**Template:** [NTU-Thesis-LaTeX-Template](NTU-Thesis-LaTeX-Template/) — `main.tex` uses 4 chapters + front matter (abstract, acknowledgement, denotation) + appendices. This plan maps content onto that skeleton.

---

## Status (as of 2026-04-16)

**First-draft thesis content has been written into the NTU template.** This plan is now the intent/outline document; the LaTeX files below are the source of truth for the actual prose.

| Component | File | State |
|---|---|---|
| Metadata / packages | [NTU-Thesis-LaTeX-Template/ntusetup.tex](NTU-Thesis-LaTeX-Template/ntusetup.tex) | drafted — author/ID/advisor still placeholder |
| Abstract (ZH + EN) | [front/abstract.tex](NTU-Thesis-LaTeX-Template/front/abstract.tex) | drafted |
| Acknowledgement | [front/acknowledgement.tex](NTU-Thesis-LaTeX-Template/front/acknowledgement.tex) | placeholder — write last |
| Denotation | [front/denotation.tex](NTU-Thesis-LaTeX-Template/front/denotation.tex) | drafted |
| Ch. 1 Introduction | [contents/chapter01.tex](NTU-Thesis-LaTeX-Template/contents/chapter01.tex) | drafted |
| Ch. 2 Background | [contents/chapter02.tex](NTU-Thesis-LaTeX-Template/contents/chapter02.tex) | drafted |
| Ch. 3 Method | [contents/chapter03.tex](NTU-Thesis-LaTeX-Template/contents/chapter03.tex) | drafted |
| Ch. 4 Experiments | [contents/chapter04.tex](NTU-Thesis-LaTeX-Template/contents/chapter04.tex) | drafted — contains [TBD] markers pending eval runs |
| Ch. 5 Conclusion | [contents/chapter05.tex](NTU-Thesis-LaTeX-Template/contents/chapter05.tex) | drafted |
| Appendix A (hyperparams) | [back/appendix01.tex](NTU-Thesis-LaTeX-Template/back/appendix01.tex) | drafted |
| Appendix B (reproducibility) | [back/appendix02.tex](NTU-Thesis-LaTeX-Template/back/appendix02.tex) | drafted |
| References | [back/references.bib](NTU-Thesis-LaTeX-Template/back/references.bib) | drafted (~20 entries) |
| Root | [main.tex](NTU-Thesis-LaTeX-Template/main.tex) | fixed duplicate `\input{chapter03}`, added chapter05 |

**Outstanding before submission:**
1. Fill author, student ID, advisor, department fields in [ntusetup.tex](NTU-Thesis-LaTeX-Template/ntusetup.tex).
2. Replace `[TBD]` markers in [chapter04.tex](NTU-Thesis-LaTeX-Template/contents/chapter04.tex) with final eval numbers from the 2022 validation runs.
3. Write real acknowledgements.
4. Compile with `xelatex → bibtex → xelatex → xelatex` and fix any template/natbib issues.

The remaining sections of this document (Front Matter, Chapter 1–5, Appendices, Writing order) describe the original *plan*; refer to the LaTeX files for the actual current text.

---

**Scope (locked for v1):**
- Teacher: pretrained EDM StormCast at [exp_3_dif_L_24_H_4_train_2_5_years](exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus) + regression [StormCastUNet.0.14000.mdlus](exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.14000.mdlus)
- Dataset: [zarr_exp3_L_24_H_24_train_2_5_years_full](exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full/) — LowRes (24ch) → HighRes (t2m, u10, v10, **qpepre**), 224×128, Taiwan domain, hourly, train 2019-08→2021-12, valid 2022 full year.
- Methods: **Progressive Distillation (PD)** and **Consistency Distillation (CD)**. ADD is out of scope for v1 (mention only as future work).

---

## Front Matter

### `front/abstract.tex`
- 1 paragraph problem: regional convective forecasting needs diffusion ensembles; teacher needs 25+ NFE per hourly step → infeasible for operational rollouts on Taiwan domain.
- 1 paragraph approach: distill the EDM teacher into a few-step student via PD and CD; adapt losses for the sparse heavy-tailed `qpepre` channel.
- 1 paragraph results: NFE reduction (25→{1,2,4,8}), per-channel RMSE parity vs teacher, precipitation CSI/FSS at operational thresholds, 12h rollout stability.
- Keywords: diffusion distillation, consistency models, progressive distillation, convective forecasting, StormCast, Taiwan.

### `front/denotation.tex`
Symbol list — draft now, finalize at the end:
- `X_t` HighRes target, `S_t` LowRes state, `M_t = F_ξ(X_{t-1}, S_t, I)` regression mean, `R_t = X_t − M_t` residual, `c` conditioning bundle, `I` invariants (lsm, orog).
- EDM: `σ ∈ [σ_min, σ_max]`, `σ_data`, `ρ`, denoiser `D_θ(x; σ, c)`, `F_θ` inner net, skip/out scalings `c_skip, c_out, c_in, c_noise`.
- Distillation: teacher `D_ψ`, student `D_θ`, PD schedule `N → N/2`, CD discretization `N(k)`, EMA decay `μ_k`, Pseudo-Huber constant `c`.
- Metrics: RMSE, MAE, CSI@τ, FSS@τ, 1D radial PSD `P(k)`.

### `front/acknowledgement.tex`
Placeholder — write last.

---

## Chapter 1 — Introduction (`contents/chapter01.tex`)

**Goal:** motivate and scope the thesis in ≤12 pages. Frames the problem from "why weather forecasting is hard" → "why ML" → "why diffusion" → "why distillation".

1.1 **Weather forecasting and its societal stakes.** Short intro: typhoon impact on Taiwan, flash-flood and orographic rainfall, operational decision timescales (0–12h nowcasting vs medium-range). Why convective-scale (km-scale, hourly) matters for the Taiwan domain specifically.

1.2 **A brief history of weather prediction methods.**
  - (a) **Numerical Weather Prediction (NWP).** Primitive equations, finite-volume/spectral solvers, operational systems (IFS, GFS, WRF/RWRF). Strengths: physical consistency, well-understood uncertainty via perturbed ensembles. Weaknesses: compute cost of convection-permitting runs, parameterization error, latency.
  - (b) **Statistical and classical ML post-processing.** MOS, analog methods, random forests for downscaling and bias correction — set the stage for learned approaches.
  - (c) **Deep learning era.** CNN/U-Net nowcasting (MetNet, DGMR), graph-based global models (GraphCast, Pangu-Weather, FourCastNet), regional convection-permitting ML (StormCast). Why these now compete with operational NWP at a fraction of the inference cost.
  - (d) **Generative forecasting.** Determinstic ML models produce blurry fields and underestimate extremes; generative models (GANs, diffusion) recover sharpness and calibrated spread — motivates the diffusion turn.

1.3 **Diffusion models for weather — intuition first.** What a diffusion model *is* in 2 paragraphs, without equations: iterative denoising from Gaussian noise, learns a score/denoiser, samples by reversing a noising process. Why this matches weather: naturally probabilistic (ensemble for free), sharp samples, conditionable on coarse state. Concrete example: GenCast (global), StormCast (regional convective).

1.4 **The inference-cost problem.** Teacher EDM sampler needs ~25 NFE per hourly step. For autoregressive rollouts (24h × 25 NFE) and ensembles (50 members × 24h × 25 NFE = 30,000 forward passes), this becomes operationally infeasible on a Taiwan-sized domain. Distillation compresses sampling to 1–8 NFE while preserving fidelity — the central problem of this thesis.

1.5 **Thesis contributions.**
  - (i) Adaptation of **Progressive Distillation** (Salimans & Ho 2022) to the StormCast two-stage regression+residual architecture on a 4-channel regional target.
  - (ii) Adaptation of **Consistency Distillation** (Song et al. 2023) with channel-weighted Pseudo-Huber + radial-PSD regularizer on `qpepre`.
  - (iii) A precipitation-aware evaluation suite (CSI/FSS/tail quantiles/spectral) over the 2022 Taiwan validation year.
  - (iv) Empirical comparison PD vs CD across NFE budgets and rollout horizons, with guidance on when to prefer each method.

1.6 **Thesis outline.**

---

## Chapter 2 — Background & Related Work (`contents/chapter02.tex`)

**Goal:** self-contained theory + lit-review chapter. Source papers live in [papers/md/](papers/md/). Structured as **(A) weather prediction landscape** → **(B) diffusion theory** → **(C) distillation theory** → **(D) evaluation**. Reader should be able to understand Ch. 3 from this chapter alone.

### A. Weather prediction: from NWP to data-driven models

2.1 **Numerical Weather Prediction (NWP) fundamentals.** Primitive equations, dynamical core vs physical parameterizations, convection-permitting resolution (~km-scale), ensemble prediction systems (EPS) for uncertainty, operational constraints (wall-clock, compute, latency). Position WRF/RWRF as the source of this thesis's training targets.

2.2 **Data-driven global weather models.** Review in chronological order with architectural contrast:
  - **FourCastNet** (AFNO, spectral). First to match IFS at synoptic scales.
  - **Pangu-Weather** (3D transformer, hierarchical time). Medium-range determinstic.
  - **GraphCast** (mesh GNN, autoregressive). Operational-quality 10-day forecasts.
  - Takeaway: determinstic ML models are fast but blurry — underestimate extremes and lose small-scale variance.

2.3 **Regional and convective-scale ML forecasting.** Why global models miss convection, orographic effects, and typhoon inner-core structure. MetNet-1/2/3 (nowcasting), DGMR (GAN-based radar nowcasting), and **StormCast** (convection-permitting, CONUS originally). Explain StormCast's two-stage regression + residual-diffusion decomposition and why it fits regional convective targets. Cite [stormcast.md](papers/md/stormcast.md). Note the CONUS→Taiwan (99ch→4ch) adaptation used in this thesis.

2.4 **Generative forecasting and ensembles.** Why probabilistic outputs matter for precipitation and extremes: reliability diagrams, spread-skill, CRPS. Transition from GAN nowcasters (DGMR) to diffusion-based forecasters. Introduce **GenCast** as the flagship global diffusion forecaster — how it frames forecasting as conditional diffusion, ensemble generation by seed variation, calibration results. Cite [gencast.md](papers/md/gencast.md). Bridge to "diffusion is the right tool, but it's slow".

### B. Diffusion models — theory

2.5 **Denoising diffusion primer.** DDPM forward/reverse chains; variance-preserving vs variance-exploding; score-matching interpretation; connection to SDEs (Song et al. 2021). Keep this tight — one page with one figure.

2.6 **Score-based SDEs and the probability-flow ODE.** Forward SDE `dx = f(x,t)dt + g(t)dw`, reverse-time SDE, deterministic PF-ODE. Why the PF-ODE matters for distillation: it defines a trajectory `x(σ)` that students try to short-circuit.

2.7 **EDM (Karras et al. 2022).** The parameterization this thesis uses end-to-end. Cover:
  - Noise schedule `σ ∈ [σ_min, σ_max]` with `ρ`-spaced discretization.
  - Preconditioning `(c_skip, c_out, c_in, c_noise)` and why it stabilizes training across σ.
  - Loss weighting and the `σ_data` choice.
  - **Heun 2nd-order sampler** — the sampler the teacher was trained to emulate and the one the CD operator Φ must match.
  - Cite [edm_karras.md](papers/md/edm_karras.md). This section is load-bearing for Ch. 3.

2.8 **Conditioning in diffusion models.** Classifier-free guidance vs concatenation conditioning. StormCast uses concatenation: `c = concat(X_{t−1}, S_t, M_t, I)`. Explain why no CFG in this setup.

### C. Distillation of diffusion models

2.9 **Why distill?** Teacher needs ~25 NFE per step; autoregressive rollouts and ensembles multiply this. Survey the solution space:
  - **Faster samplers** (DDIM, DPM-Solver, EDM Heun): reduce NFE from 1000→25 but plateau.
  - **Distillation**: train a student to match teacher outputs in fewer steps — the approach of this thesis.
  - **Direct one-step generators** (rectified flow, shortcut models): mentioned briefly.

2.10 **Progressive Distillation (Salimans & Ho 2022).** Algorithm 2 in detail: pair two teacher DDIM steps, compute the implied denoised `x̃`, train student to match it in one step, halve the step count, repeat. `v`-prediction parameterization and why it stabilizes low-NFE regimes. Cite [progressive_distillation.md](papers/md/progressive_distillation.md).

2.11 **Consistency Models and Consistency Distillation (Song et al. 2023).** Define the consistency function `f: (x_σ, σ) → x_{σ_min}` with boundary condition `f(x, σ_min)=x`. Training via CD: sample `(x_σ, x_{σ'})` pairs along the PF-ODE using a teacher Φ operator (Heun step), enforce `f(x_σ, σ) ≈ f_EMA(x_{σ'}, σ')`. Pseudo-Huber loss, EMA target net with `μ_k = μ_0^(N_0/N_k)`, adaptive discretization `N(k)` (sqrt growth). 1-step and multi-step sampling. Cite [consistency_model.md](papers/md/consistency_model.md).

2.12 **Adversarial Diffusion Distillation (brief).** Sauer et al. — hinge GAN + score-distillation, scaffolded in the codebase but deferred to future work. Cite [adversarial_distillation.md](papers/md/adversarial_distillation.md).

2.13 **Prior applications of distillation to weather/physics.** This is a *thin* literature — cover what exists: distilled DGMR variants, any CorrDiff/StormCast distillation precedents, and adjacent work on diffusion surrogates for PDEs. Position this thesis as (to our knowledge) the first PD+CD study on a regional convection-permitting diffusion forecaster.

### D. Evaluation

2.14 **Evaluation of precipitation forecasts.** Categorical scores (CSI, FSS, frequency bias) at operational thresholds; spectral diagnostics (radial PSD) for sharpness; rollout error growth for autoregressive drift; CRPS and rank histograms for ensembles. Justify why RMSE alone is misleading for `qpepre`.

---

## Chapter 3 — Method (`contents/chapter03.tex`)

**Goal:** describe exactly what was built. Reference the actual code, not idealized pseudocode.

3.1 **Problem setup.** Given frozen teacher `D_ψ` (EDMPrecond SongUNet) and regression `F_ξ` (StormCastUNet), learn student `D_θ` s.t. `x̂_student ≈ x̂_teacher` with ≤ K NFE.

3.2 **Dataset and preprocessing.**
  - Zarr layout (LowRes/HighRes/invariants), 24ch → 4ch, 224×128 Taiwan grid.
  - Per-channel normalization stats: HighRes means `[295.98, −1.58, −2.65, 0.182]`, stds `[5.61, 3.84, 5.66, 9.12]`.
  - `qpepre` analysis: wet-fraction, heavy-tailed histogram, justification for per-channel loss weighting.
  - Loader: [data_loader_rwrf_era5_stable.py](stormcast/data_loaders/data_loader_rwrf_era5_stable.py) (verify path).

3.3 **Conditioning construction.** `c = concat(X_{t-1}, S_t, M_t, I)`. Helper [stormcast/utils/nn.py:80](stormcast/utils/nn.py#L80) (`build_network_condition_and_target`). Diagram.

3.4 **Teacher recap.** EDM Karras `σ_min=0.002, σ_max=80, σ_data=0.5, ρ=7`, 25-step Heun sampling. Factory at [stormcast/utils/nn.py:25](stormcast/utils/nn.py#L25).

3.5 **Method A — Progressive Distillation.**
  - Algorithm (Salimans–Ho Alg. 2) in denoised-prediction space `x̃`.
  - Per-phase loop `N → N/2`, student promoted to teacher between phases.
  - Implementation: [train_progressive.py](stormcast/train_progressive.py), [trainer_progressive.py](stormcast/utils/trainer_progressive.py) (`progressive_distillation_loop`), loss at [progressive_distillation_loss.py](physicsnemo/experimental/metrics/diffusion/progressive_distillation_loss.py), config [progressive.yaml](stormcast/config/progressive.yaml).
  - Inference: `progressive_distilled_forward` at [stormcast/utils/nn.py:180](stormcast/utils/nn.py#L180).
  - `qpepre` handling: Pseudo-Huber on residual space; tail-quantile monitoring per phase (wet-fraction, P95, P99).

3.6 **Method B — Consistency Distillation.**
  - Consistency parameterization with boundary `f(x, σ_min)=x` via `ConsistencyPrecond`.
  - Loss: [consistency_loss.py](physicsnemo/experimental/metrics/diffusion/consistency_loss.py) — Pseudo-Huber (`c=5.4e-4`), channel weights `β_k`, radial log-PSD L1 on `qpepre`.
  - EMA target: [ema.py](stormcast/utils/ema.py), decay `μ_k = μ_0^(N_0/N_k)`.
  - Adaptive discretization `N(k)` sqrt growth `N_0=2 → N_total=150`.
  - Teacher Φ = **Heun** step (not Euler — matches sampler teacher emulates).
  - 1-step inference sampling `x ~ N(0, σ_max² I)` → `consistency_model_forward`; inference weights from `ema_state.pt`.
  - Implementation: [train_consistency.py](stormcast/train_consistency.py), [trainer_consistency.py](stormcast/utils/trainer_consistency.py), configs [consistency.yaml](stormcast/config/consistency.yaml) + [training/consistency.yaml](stormcast/config/training/consistency.yaml).

3.7 **Design choices specific to 4-channel RWRF.**
  - Per-channel `β_k` weighting with `qpepre` upweight.
  - Radial log-PSD regularizer on `qpepre` to prevent high-frequency collapse.
  - Optional `log1p`/`asinh` transform on `qpepre` — to be tested.
  - Exact copy of teacher EDM hyperparameters (silent failure mode otherwise).

3.8 **Training infrastructure.** `torchrun` multi-GPU, Hydra overrides, auto-resume (`_detect_resume_state`), checkpoint layout `StormCast_<method>/<experiment>/0/…`.

---

## Chapter 4 — Experiments & Results (`contents/chapter04.tex`)

**Goal:** empirical answer to "does distillation work on this dataset, and which method wins".

4.1 **Experimental protocol.**
  - Validation year: 2022 (8,760 hourly samples). Override stale shipped `valid_dates`.
  - NFE budgets evaluated: teacher-25 baseline; students at 1, 2, 4, 8.
  - Hardware, wall-clock, seeds.

4.2 **Metrics.**
  - Deterministic: per-channel RMSE/MAE (de-normalized) for `{t2m, u10, v10, qpepre}`.
  - Precipitation: CSI, FSS, frequency bias at τ ∈ {0.1, 1, 5, 10, 20} mm/h.
  - Spectral: radial log-PSD of `qpepre`, student vs teacher.
  - Rollout: 1h / 3h / 6h / 12h RMSE and CSI — autoregressive drift.
  - (Optional) 10-member CRPS and rank histograms for `qpepre`.k j

4.3 **Results — Progressive Distillation.** Per-phase loss curves, NFE vs error Pareto, wet/dry distribution drift, heatmaps + PSDs for representative storm cases.

4.4 **Results — Consistency Distillation.** 1-step vs 2-step vs 4-step (multistep CD), EMA-vs-online ablation, spectral regularizer ablation, channel-weight ablation.

4.5 **PD vs CD comparison.** Same NFE budgets, same validation splits. Tables + Pareto plot.

4.6 **Qualitative cases.** Pick 3–4 Taiwan weather events in 2022 (typhoon landfall, frontal convection, orographic rain, dry day) — show teacher vs PD vs CD at matched NFE.

4.7 **Failure modes.** Precipitation mode collapse monitoring, tail-quantile regression, rollout divergence cases.

4.8 **Discussion.** When to prefer PD vs CD; cost of training; sensitivity to `qpepre` transform; implications for operational ensembles.

---

## Chapter 5 — Conclusion (add as 5th chapter; update `main.tex`)

> Note: the template ships with 4 chapters and a duplicate `\input{contents/chapter03}` line in [main.tex](NTU-Thesis-LaTeX-Template/main.tex) that should be fixed; add `chapter05.tex` for conclusion.

5.1 Summary of contributions.
5.2 Limitations: single teacher checkpoint, regional domain only, no ADD baseline, no ensemble metrics if skipped.
5.3 Future work: ADD (already scaffolded — [train_add.py](stormcast/train_add.py)), log/asinh `qpepre` transform, distillation of longer rollouts, ensemble calibration, transfer to other regional domains.

---

## Appendices (`back/appendix01.tex`, `appendix02.tex`)

- **A. Hyperparameter tables.** PD phases, CD schedules, optimizer, batch sizes, learning rates.
- **B. Extended metric tables.** Per-channel × per-NFE × per-horizon.
- **C. Additional qualitative panels.** More 2022 case studies.
- **D. Reproducibility.** Exact commands, config overrides, checkpoint paths, commit hashes.

---

## `back/references.bib` — key entries to add

- Karras et al. 2022 (EDM)
- Salimans & Ho 2022 (Progressive Distillation)
- Song et al. 2023 (Consistency Models)
- Song & Dhariwal 2023 (Improved CT)
- Pathak et al. / Price et al. (StormCast)
- Price et al. 2024 (GenCast)
- Sauer et al. 2023 (ADD, future-work citation)
- Ho et al. 2020 (DDPM), Song et al. 2021 (score SDE)
- Roberts & Lean 2008 (FSS), Schaefer 1990 (CSI)

---

## Writing order (recommended)

1. Ch. 3 Method — write while code is fresh; pull figures straight from training runs.
2. Ch. 4 Experiments — write alongside the final eval runs, not after.
3. Ch. 2 Background — once methods are frozen so notation matches Ch. 3.
4. Ch. 1 Intro + Ch. 5 Conclusion — last.
5. Abstract + acknowledgements — after Ch. 1/5.
6. Denotation — finalize once all notation is locked.

## Open questions to resolve before writing Ch. 3

- Does the teacher apply a `log/asinh` transform to `qpepre` internally? (Gotcha from CLAUDE.md — check loader + preprocessing.)
- Confirm CD is using **Heun** Φ step in current code, not Euler.
- Decide whether ensemble metrics (CRPS, rank histogram) are in v1 or deferred.
- Confirm PD phase schedule `N: 16→8→4→2→1` vs `32→…→1` for the final runs.
