# Thesis to-do — experiments & data still needed

Thesis has shifted to **FlowCast (Conditional Flow Matching)** as the headline
method, with the EDM diffusion residual kept as the reference baseline. The
LaTeX in `NTU-Thesis-LaTeX-Template/` has been rewritten accordingly
(distillation chapters removed; FlowCast method + real results added). This
file lists what is still missing to turn the **provisional** results into a
defensible thesis claim.

Current evidence base (all already in the repo and now cited in Ch. 4):
- 3-way rollout scoreboard at the **~2 M-sample** budget, denormalised /
  mm·h⁻¹: `experiment_scripts/results/main_experiment/`.
- Single-step **standardised** validation RMSE (log1p & qpw ablations):
  `experiment_scripts/result_table.md`.
- Trained checkpoints: EDM `EDMPrecond.0.31000` (and on to `0.73000`),
  FlowCast `FlowCastPrecond.0.20000` … canonical `0.140000` + `ema_state.pt`.

---

## 1. Convergence / matched-budget re-score  *(highest priority)*

The headline table is a **single snapshot at ~2 M samples**, where adjacent
validation logs drift 10–30 %. The claim "FlowCast ≈ diffusion at 1/3 the cost"
needs a converged, noise-averaged comparison.

- [ ] Re-run `run_main_experiment.sh` with **both heads at their highest
      matched budget**. FlowCast canonical is at step 140 000 (~13.4 M samples);
      the EDM baseline currently only reaches step 73 000 (~4.67 M samples) →
      **train the EDM baseline further** so the two can be compared at a genuinely
      matched, converged sample budget (or report at the largest common budget
      and say so explicitly).
- [ ] Average each metric over **~5 neighbouring validation snapshots** (or
      bootstrap over validation hours) and report a spread/confidence interval,
      so sub-5 % differences are not over-read.
- [ ] Regenerate every Ch. 4 figure at the converged budget (currently the
      figures are the ~2 M-sample versions copied into
      `figures/results/` and `figures/flowcast/`).

## 2. Low-step FlowCast NFE sweep  (K = 1, 2, 4)

The "straight-line ⇒ cheap" thesis is strongest at the extreme low-step limit,
but the scoreboard only has K ∈ {10, 15, 20}. The sweep harness exists
(`experiment_scripts/flowcast_nfe_sweep.py`) but **no `nfe_sweep.csv` has been
produced**.

- [ ] Run `run_flowcast_nfe_sweep.sh` to get quality + wall-clock for
      K ∈ {1, 2, 4, 8, 10, 15, 20(, …50)} on a fixed FlowCast checkpoint.
- [ ] Produce the **NFE–quality Pareto plot** (RMSE/CRPS/CSI vs NFE and vs
      wall-clock) and add it to §4.4 (currently that section argues from the
      three-point scoreboard only).
- [ ] Report the EDM baseline at a couple of reduced step counts (e.g. Heun
      N = 8, 4) on the same axes, to show the diffusion sampler's steeper
      degradation below ~20 NFE.

## 3. Close the heavy-rain CSI gap

FlowCast's weakest metric is pixel-exact CSI at 5–10 mm·h⁻¹ (it is more
conservative / lower-FAR). Needs targeted ablations, not just more training.

- [ ] Tail-aware loss on `qpepre`: explicit tail-quantile or focal term;
      compare CSI/FAR at 5/10/16/20 mm·h⁻¹.
- [ ] Re-test higher `w_pr` (2.2, 2.4) **together with** the spectral term at a
      converged budget — the current qpw sweep is single-step standardised only.
- [ ] Try a stronger dynamic-range transform (`asinh`) vs `log1p` for qpepre.
- [ ] Add threshold = 20 mm·h⁻¹ to the categorical tables (currently top out at
      16); typhoon-core rain lives above 20.

## 4. Spectral diagnostics (quantitative, not just training PNGs)

§4.9 shows one training-time qpepre spectrum. The thesis claims "sharp, tracks
the observation spectrum" but has **no validation-set radial log-PSD
comparison plot** (diffusion vs FlowCast vs truth).

- [ ] Emit a validation-averaged radial log-PSD CSV for qpepre from
      `compare_diffusion_vs_flowcast.py` / `_eval_utils.py` and plot
      diffusion vs FlowCast vs ground truth on one axis.

## 5. Ensemble / probabilistic calibration

CRPS is from a small per-hour ensemble. A cheap sampler's main payoff is large
ensembles — currently unexploited.

- [ ] 50–100-member FlowCast ensemble: CRPS, **rank histogram**, spread–skill
      ratio for qpepre. Compare against a (necessarily smaller) diffusion
      ensemble at equal wall-clock.

## 6. Qualitative case studies on named 2022 events

§4.10 panels are currently generic validation sequences at rollout step 0.
The thesis wants event-driven cases.

- [ ] Pick concrete 2022 dates and regenerate **mid-rollout** panels (truth |
      diffusion | FlowCast) for: a typhoon landfall; a Mei-yu frontal-convection
      day (May/Jun); an orographic-rain day over the Central Mountain Range; a
      dry quiescent day (false-alarm test). Need the date list + a panel run at
      those `t0` indices.

## 7. Rollout robustness

- [ ] Longer rollout (24 h) to confirm the `t2m` drift seen after ~8 h
      (§4.7) does not become a stability problem, and to get the 24-h numbers
      the operational-cost argument in §1.4 / §4.4 assumes.
- [ ] Multi-seed rollouts to put error bars on the rollout-RMSE curves.

## 8. Thesis-asset / bookkeeping gaps

- [ ] Fill placeholders in front matter: advisor name, student ID, oral-exam
      date, DOI (`ntusetup.tex`).
- [ ] EDM-baseline loss curve figure (only the FlowCast one is in
      `figures/flowcast/`).
- [ ] Decide whether to keep the legacy 3-way leg (§4.8) given the grid/FoV
      mismatch caveat, or demote it to an appendix.
- [ ] Update `slide_content.md` / `thesis_defense.pptx` — they still describe
      the old distillation framing (PD/CD/ADD) and need to match the FlowCast
      thesis.
- [ ] Confirm the cleaned-store qpepre log-space std (used as ≈0.37 in
      App. A / §4.6) from `HighRes/stats/stds.npy` and pin the exact value.

## 9. Optional / stretch

- [ ] Distilled few-step **diffusion** comparator, to answer "could a distilled
      diffusion student match FlowCast's speed while keeping its heavy-rain CSI?"
      (the one comparison the current thesis explicitly does not make).
- [ ] Stochastic-interpolant / bridge variant anchored at the regression mean
      (see `bridge.md`) as an alternative short-transport flow.
- [ ] Transfer test on a second domain / regression mean (e.g. HRRR StormCast).
