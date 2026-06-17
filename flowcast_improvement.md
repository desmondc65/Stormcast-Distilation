# FlowCast Improvement Proposal

**Goal.** Turn the current "swap EDM diffusion for I-CFM" baseline into a thesis-defensible methodological contribution by reshaping *how the flow itself is constructed* — not just which solver runs on it. Each direction below has a concrete novelty story, a measurable performance hypothesis, and implementation pointers into the existing code.

**Current state (baseline to beat).**
- Method: standard I-CFM (Lipman 2023 / Tong 2024) on the standardized residual `r = M_{t+1} - μ_{t+1}`. Endpoints `x_0 ~ N(0, I)`, `x_1 = r / σ_data`, path `x_t = (1-t) x_0 + t x_1 + σ_path ε`, target `u_t = x_1 - x_0`. Loss is per-pixel MSE plus a per-channel β-weight and an optional log-PSD penalty on qpepre. Sampler is fixed-step Euler / Midpoint.
- Headline result: ~equal to slightly better RMSE than EDM teacher across `t2m, u10, v10, qpepre`, ~3× faster inference (10 Euler steps vs 18 Heun steps × 2 NFE).
- Code lives in [stormcast/utils/flowcast_loss.py](stormcast/utils/flowcast_loss.py), [stormcast/utils/flowcast_precond.py](stormcast/utils/flowcast_precond.py), [stormcast/utils/trainer_flowcast.py](stormcast/utils/trainer_flowcast.py), [stormcast/utils/nn.py:163](stormcast/utils/nn.py#L163).

**The advisor's actual ask.** A method swap is engineering, not research. The thesis needs an idea that *exploits something specific about the residual NWP setup* — the regression mean μ, the heavy-tailed precipitation channel, the multi-scale spectrum, the autoregressive rollout — to make the flow itself work better. The five directions below all do that. They are not mutually exclusive; the recommended thesis bundles two or three of them.

---

## Direction 1 (recommended primary) — Anchored Stochastic-Interpolant Bridge: μ → M, not noise → residual

### Idea
Drop the "Gaussian noise to residual" framing entirely. The frozen regression mean μ already provides a physically-meaningful approximation of `M_{t+1}`. Instead of throwing that information away and re-generating from `N(0, I)`, **let the regression sample itself be the prior endpoint of the flow**, and learn a stochastic interpolant from `μ` to the ground-truth `M`.

Concretely, build the path as

```
x_0 = μ + σ_prior · ε_0           (regression mean + small Gaussian)
x_1 = M_{t+1}                     (the truth, in raw physical / log1p space)
x_t = α(t)·x_0 + β(t)·x_1 + γ(t)·ε
```

with `α(0)=1, β(0)=0, α(1)=0, β(1)=1` and a `γ(t)` that vanishes at both endpoints (Albergo & Vanden-Eijnden 2023 stochastic-interpolant family; closely related to I²SB / direct Schrödinger Bridge Matching, Liu et al. 2023). The student regresses the velocity `u_t = ∂_t x_t` exactly as in I-CFM. At inference you initialize the ODE at `μ + σ_prior ε`, not at white noise — the flow is *short* because the endpoints are close.

### Why this is a genuine contribution
1. **It changes the geometry of the problem.** In standard FlowCast the ODE has to transport mass across a Wasserstein-large gap (Gaussian → residual manifold), and uses up its straight-line budget doing so. With μ as the prior, the gap is exactly the residual `r`, but parameterized as a *bridge* with two-sided endpoint constraints rather than a one-sided drift. This is the I²SB framing applied to regional NWP residuals — to my knowledge unpublished.
2. **It uses the regression mean as more than conditioning.** Today μ is just another channel concatenated into the SongUNet input. The bridge formulation makes μ a *structural* part of the generative process: at t=0 the model is already at μ, and the only thing it has to learn is how to drift toward the truth. This should produce better-calibrated samples at low NFE because step 1 already has the right *bulk* structure (large-scale temperature, mean wind), and the remaining steps only sharpen detail.
3. **It naturally regularizes the autoregressive rollout.** With Gaussian-prior I-CFM, every rollout step injects fresh white-noise variance and can drift. With an anchored prior the variance injected per step is bounded by `σ_prior`, so 6-12 h rollout RMSE should degrade less. This is testable on the existing [experiment_scripts/plot_rollout_12h.py](experiment_scripts/plot_rollout_12h.py).

### Concrete schedule
A clean choice (and what I'd default to in ablation):
```
α(t) = 1 - t,    β(t) = t,    γ(t) = σ_max · sqrt(t·(1-t))
```
which gives a Brownian-bridge probability path between `x_0` and `x_1` with maximum noise injection at `t=1/2`, and `γ(0) = γ(1) = 0`. The target field becomes
```
u_t = x_1 - x_0 + σ_max · (1-2t)/(2·sqrt(t(1-t))) · ε
```
This bridge target is unbiased even though it depends on ε (Albergo 2023, App. B). Implement it carefully — the `ε`-dependence is what gives the bridge non-trivial generative capacity rather than collapsing to a deterministic residual head.

For autoregressive stability, also offer the simpler **deterministic bridge** ablation (γ ≡ 0): `x_t = (1-t)μ + t M`, target `u_t = M - μ = r`. That degenerates to "regress the residual at noise level γ(t)=0" — useful as a sanity check that the *stochasticity*, not just the anchoring, is what helps.

### Implementation pointers
- New class `BridgeMatchingLoss` in [stormcast/utils/flowcast_loss.py](stormcast/utils/flowcast_loss.py), called from [trainer_flowcast.py:224-231](stormcast/utils/trainer_flowcast.py#L224-L231). Inside the loss, you already have `target = M_{t+1} - μ_{t+1}` because `build_network_condition_and_target` subtracts μ at [stormcast/utils/nn.py:131](stormcast/utils/nn.py#L131). You need to *not* subtract it for this method — add a `condition_list="anchored"` mode that returns `(condition, M_{t+1}, μ_{t+1})` separately so the loss can construct the bridge.
- A new sampler `bridge_model_forward` next to [flowcast_model_forward](stormcast/utils/nn.py#L163), initialised at `z = μ + σ_prior · ε` and integrated to t=1 directly (no `* sigma_data` rescale at the end because we are operating in raw space).
- Hydra config: add `stormcast/config/training/bridge.yaml` with knobs `prior: "regression"`, `sigma_prior`, `sigma_max`, and a path-type enum.

### Risk + mitigation
- The ε-dependent target is noisier than the I-CFM target. Mitigate by EMA on the student (already in code), and by averaging multiple ε draws per sample at low cost (`K=2` is cheap).
- Verify that the bridge's empirical marginal at t=1 still matches the data — easy unit test: sample from the bridge with the *true* `u_t` (no neural net) and check that you recover `M_{t+1}`.

---

## Direction 2 — Heavy-tailed / channel-aware prior for qpepre

### Idea
The qpepre channel is the make-or-break of this thesis (CLAUDE.md §8). It's sparse, heavy-tailed, and even after log1p its standardised distribution has fat tails that a unit-Gaussian prior poorly approximates. **Choose a prior matched to the data.** Two concrete options:

1. **Per-channel Gaussian with channel-specific variance.** Keep `x_0 ~ N(0, diag(s_1², …, s_C²))` where `s_c` is fit on the empirical residual std per channel. Cheap, deterministic, and theoretically clean — a stochastic interpolant only requires the prior to be a tractable distribution.
2. **Heavy-tailed prior on qpepre only.** Replace the qpepre slice of `x_0` with samples from a Student-t (ν≈4-6) or a Laplace, both of which admit reparameterisation gradients and CDF inversion. Empirically the qpepre residual matches a Student-t better than a Gaussian. The flow then transports a heavy-tailed prior to a heavy-tailed target — a *shorter* OT distance than Gaussian→heavy-tailed.

### Why this is novel
Most flow-matching papers assume `N(0, I)` because it's the path of least resistance. There's a thin body of work on non-Gaussian priors for flow matching (e.g. for protein structures), but for atmospheric science the assumption has gone uncontested. Showing that a heavy-tailed prior recovers extreme precipitation events better — measured by CSI/HSS at the 20 mm/h threshold, which CLAUDE.md §7 already lists as the headline precip metric — is a clean novelty story.

### Implementation pointers
- A `Prior` interface (`sample(shape, device)`, `log_prob(x)`) injected into `FlowCastLoss` and `flowcast_model_forward`. Default `GaussianPrior(σ=1)` for parity.
- For Student-t: `torch.distributions.StudentT(df=ν).rsample(shape)`. Standardize so that the prior std matches `σ_data` per channel.
- Sweep ν ∈ {3, 5, 7, ∞} as a per-channel ablation; expect the optimum around 5 based on the qpepre tail behaviour.

### Risk + mitigation
- Student-t with `ν ≤ 2` has infinite variance and will destabilise training. Clamp `ν ≥ 3` and the prior to a finite quantile range during sampling (e.g. `clip(|x_0|, q=0.999)`).

---

## Direction 3 — OT-CFM with mini-batch optimal coupling

### Idea
I-CFM (the FlowCast baseline) pairs each data sample `x_1` with an *independent* noise sample `x_0`. Tong et al. 2024 (already cited in your flow_cast.md) shows that pairing them with a *minibatch optimal transport* coupling produces straighter trajectories that need fewer Euler steps. This is OT-CFM.

For your setup: inside each minibatch of B residuals `{r_i}`, sample B noise vectors `{z_j}`, solve the small B×B linear-assignment problem `min_π Σ ‖z_{π(i)} - r_i‖²` (Hungarian or Sinkhorn), then build the path between coupled `(z_{π(i)}, r_i)` pairs instead of random `(z_j, r_i)`.

### Why this is a contribution
- Applied to NWP residuals this is unpublished. The FlowCast paper explicitly *did not* use OT coupling.
- OT-CFM is what enables the 1-step-quality story (`CFM at 1 NFE ≈ 10 NFE` in the FlowCast paper, Table 6). On your problem, hitting acceptable 1-step quality would be a striking efficiency win and a *headline thesis result*: "diffusion needs 36 NFE, FlowCast needs 10, OT-CFM-FlowCast needs 1, at equivalent CSI."
- It composes with Directions 1 and 2 — you can do OT coupling between μ and M (bridge OT), and you can use a heavy-tailed prior on qpepre with Sinkhorn OT in standardized space.

### Implementation pointers
- In `FlowCastLoss.__call__`, after drawing `x_0`, compute the B×B cost `C[i,j] = ‖x_0[j] - x_1[i]‖²` averaged over channels & pixels, solve with `scipy.optimize.linear_sum_assignment` (or `ot.emd` from the POT library if installed), permute `x_0` accordingly.
- Per Tong 2024, the cost is `O(B³)` worst case but `B ≤ 128` is fine. Sinkhorn (`ε=0.1`) is a constant-time alternative.
- Add a hydra flag `coupling: "iid" | "ot" | "sinkhorn"`.

### Risk + mitigation
- The OT cost can dominate small fields. Compute it on a downsampled (e.g. 24×24) version of the field — coupling quality is robust to that.

---

## Direction 4 — Distillation to 1-step (Consistency / Mean-Flow / Hyper-distillation)

### Idea
The 10-Euler-step FlowCast is the new *teacher*. Distill it down to a 1-step generator that maps `(noise, condition) → r_{t+1}` in a single forward pass. Two flavours, pick one:

1. **Consistency distillation** ([papers/md/consistency_model.md](papers/md/consistency_model.md) — Song et al. 2023). Train a consistency model `f_φ(x_t, t, c)` to map any point on the FlowCast PF-ODE trajectory back to `x_1`. After convergence, `f_φ(noise, t=0, c)` gives the 1-step sample.
2. **Mean-flow / Rectified flow distillation** (Liu 2023 — generate `(noise, sample)` pairs from the FlowCast teacher, fit a new flow regressing the *straight line* between them; one round suffices for near-1-step quality on this size of problem).

### Why this is a contribution (and why it's not yet stale)
- The repo's earlier history removed CD/PD as failed branches (git log). Returning to them on a *FlowCast* teacher rather than an EDM teacher is a different experiment: the FlowCast teacher's PF-ODE is already nearly straight, so distillation should converge much faster than diffusion distillation did. That's a clear hypothesis to test.
- 1-step inference on a regional NWP system is a real-world deployment win — "ensemble of 50 members in under a second per step" enables operational ensembling that 10-step CFM can't.
- A 1-step student also closes the *autoregressive accumulation* loop: error-per-step is bounded, and you can do *longer rollouts* in fixed wallclock budget.

### Implementation pointers
- Add `train_flowcast_distill.py` that reuses [trainer_flowcast.py](stormcast/utils/trainer_flowcast.py) but loads two copies of the model: a frozen `teacher` (the trained EMA from §6.1 of CLAUDE.md) and a trainable `student` initialized from the teacher.
- Loss: pick pairs of times `t < t'`, integrate the teacher one Euler step from `x_{t'}` to `x_t`, train the student to satisfy `f_φ(x_{t'}, t') ≈ f_φ(x_t, t)`.
- Inference: 1-step Euler with the student.
- Re-use the existing EMA infra ([stormcast/utils/ema.py](stormcast/utils/ema.py)).

### Risk + mitigation
- Distillation is finicky on qpepre. The remedy is to keep the per-channel weight `qpw=2.0` from §6.3 in the distillation loss, and add the spectral log-PSD term [flowcast_loss.py:174-186](stormcast/utils/flowcast_loss.py#L174-L186) as a sharpness regularizer.

---

## Direction 5 — Physics-aware flow: divergence-free wind projection inside the ODE

### Idea
Most of the wind error in NWP residual heads comes from violating physical constraints — in particular the near-horizontal-divergence-zero structure of 10-m wind fields at synoptic scale. Inject a differentiable physical projector into the ODE integrator so that at every Euler step the sampled wind state is projected onto the manifold of low-divergence fields.

Concretely, let `P` be a fixed linear operator implementing Helmholtz decomposition and zeroing out the divergent component above some scale. At each ODE step:
```
z_{i+1} = z_i + v · dt
(z_{i+1})_wind = P · (z_{i+1})_wind   # project u10, v10 channels
```
`P` is precomputed once per resolution as a 2-D FFT mask + inverse FFT, fully differentiable, ~negligible cost.

### Why this is a contribution
- Physics-informed flow matching is barely a year old as a topic (a handful of arXiv preprints, mostly on molecular dynamics). Atmospheric application is *open*.
- The projector is **not a loss term** — it's a *constraint at integration time*, so it's guaranteed to hold at every step rather than approximately satisfied in expectation. This is a real methodological difference from the "add a physics-loss" approach that's already overdone in the literature.
- It composes with all of Directions 1-4.

### Implementation pointers
- Add `WindDivergenceProjector` in a new file `stormcast/utils/physics_projector.py`. Operates on the `(u10, v10)` slice of the state in spectral space.
- Use it in [flowcast_model_forward](stormcast/utils/nn.py#L163) inside the integration loop — one extra FFT/iFFT pair per step.
- Train *with* the projector active (so the velocity learns to be compatible with it) — this is important; if you only add the projector at inference you get a distributional mismatch.

### Risk + mitigation
- For a regional Taiwan domain (192×96) the periodicity assumption of FFT is wrong. Use a windowed FFT or pad-and-crop. This is non-trivial but tractable.
- If the constraint is too tight, performance drops on small-scale features. Make the projection *soft* (frequency-dependent damping `H(k)` rather than hard cutoff).

---

## Recommended thesis bundle

Strong CS-thesis contribution = **Direction 1 (Anchored Bridge) + Direction 3 (OT coupling) + Direction 4 (1-step distillation)**, in that priority order.

- D1 reframes the problem (the *novel idea*) — the residual stops being a noise→signal task and becomes a μ→M bridge. This is the chapter that gives the thesis its title.
- D3 is the *natural extension* that exploits D1: coupling μ-perturbations to M-samples via OT makes the bridge geometrically optimal. One paragraph of theory, one figure of trajectory straightness.
- D4 turns the result into a *deployment story* with a clean ablation: anchored bridge → OT-anchored bridge → 1-step OT-anchored bridge. This is the chapter the advisor will point to when defending the "is this enough work?" question.

Direction 2 is an easy 2-week add-on inside D1 and worth doing for the qpepre angle. Direction 5 is high-risk / high-impact — keep it as a stretch chapter or future-work appendix unless the bridge work converges fast.

### Concrete eval plan (re-uses existing infra)
1. **Standard validation grid.** Run all variants through [experiment_scripts/compare_diffusion_vs_flowcast.py](experiment_scripts/compare_diffusion_vs_flowcast.py) and stitch into the existing 3-way scoreboard (CLAUDE.md §6.6). Add columns for the new methods.
2. **Step efficiency curve.** Reuse [experiment_scripts/flowcast_nfe_sweep.py](experiment_scripts/flowcast_nfe_sweep.py) — plot CRPS / CSI vs NFE for I-CFM, OT-CFM, Bridge, 1-step student. Headline plot of the thesis.
3. **Rollout stability.** Reuse [experiment_scripts/plot_rollout_12h.py](experiment_scripts/plot_rollout_12h.py). Anchored prior should win at 6-12 h.
4. **Extreme-event detection.** CSI/HSS at 5/10/20 mm/h qpepre thresholds — the heavy-tailed-prior story (D2) shows up here.
5. **Trajectory-straightness diagnostic.** New plot: sample 10 trajectories per method, plot `‖v_θ(x_t, t) - (x_1-x_0)‖` vs `t` averaged over a validation batch. OT-CFM and the bridge should be flatter / closer to zero. This is the figure that visually sells the "geometrically better" claim.

### Compute budget estimate
- Each new training run: ~2 M samples ≈ existing 20-30 k step budget, ~3-4 days on one GPU based on current run times in `runs/`.
- Total: ~6 runs (Bridge ×2 priors, OT-Bridge, 1-step student, plus a heavy-tail-prior ablation) ≈ 3-4 weeks of GPU time. Compatible with a thesis timeline.

---

## What I would NOT do

- **Pure architecture swap** (replace SongUNet with Earthformer / DiT / Mamba). Engineering, not research; the advisor's critique applies again.
- **More channels / more conditioning** (radar nowcasts, satellite). Useful but not algorithmic. Save it for "future work."
- **Yet another loss term** (perceptual loss, GAN loss on qpepre). Reviewers see this as fishing.
- **Pure ensemble-CRPS tuning.** It's a metric story, not a method story.

The advisor's critique is fundamentally that *the generative formulation* hasn't been re-thought. Directions 1, 2, 3, 5 all re-think the formulation. Direction 4 industrialises the win. That's the thesis.
