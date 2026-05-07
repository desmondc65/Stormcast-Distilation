# EDM (Karras 2022) and Its Role in StormCast

A working note explaining what EDM is, the design choices it makes, and how
each of those choices is wired into the StormCast pipeline (and therefore
into the distillation work in this repository). Sources: [papers/md/edm_karras.md](papers/md/edm_karras.md), [papers/md/stormcast.md](papers/md/stormcast.md), and the local code under [stormcast/](stormcast/) and [physicsnemo/](physicsnemo/).

---

## 1. What problem EDM solves

Diffusion models in 2021 (DDPM, NCSN, score-SDE, iDDPM, DDIM, …) were a tangle of mutually-justifying choices: a particular forward SDE forced a particular noise schedule, which forced a particular network parameterization, which forced a particular loss weighting, which forced a particular sampler. Karras et al. argue this is unnecessary. They show that *training* and *sampling* and *preconditioning* are independent design axes, and that prior frameworks differ only because they happen to make different choices on each axis.

Their contribution is threefold:

1. A **common mathematical framework** that subsumes VP / VE / iDDPM by exposing four free functions: noise schedule σ(t), scaling s(t), discretization {tᵢ}, and four preconditioning scalars (cₛₖᵢₚ, cᵢₙ, cₒᵤₜ, cₙₒᵢₛₑ).
2. **Best-practice values** for those functions, derived from first principles (unit-variance inputs, unit-variance targets, locally linear ODE trajectories, equalised truncation error).
3. **A drop-in deterministic sampler (Heun 2nd-order)** and an **optional stochastic "churn" sampler** that match or beat prior work at a fraction of the NFE (Number of Function Evaluations).

The headline result is FID 1.79 / 1.36 on CIFAR-10 / ImageNet-64 with only 35 NFE, against ~250+ NFE for the original DDPM.

---

## 2. The unifying framework

### 2.1 Probability-flow ODE in EDM form

EDM models data corrupted by Gaussian noise of magnitude σ:

$$p(x;\sigma) = p_\text{data} * \mathcal{N}(0, \sigma^2 I)$$

Sampling = solving the probability-flow ODE backward in time:

$$dx = -\dot\sigma(t)\,\sigma(t)\,\nabla_x\log p(x;\sigma(t))\,dt$$

Karras' first simplification: choose **σ(t) = t** and **s(t) = 1**. With this choice σ and t become interchangeable, and the ODE collapses to the very clean form

$$\frac{dx}{dt} = \frac{x - D(x;\,t)}{t}$$

where D(x; σ) is the *ideal denoiser* — the L₂-optimal estimator of the clean signal y given noisy input x = y + n at noise level σ. This ODE has a beautiful geometric interpretation: at any state (x, t), the tangent points from x straight at the denoiser's clean-data estimate, so a single Euler step to t = 0 already yields the denoised image. That keeps trajectories nearly linear except in a narrow σ band where p(x; σ) actually changes shape, which is why discretisation works at low step counts.

### 2.2 Score from denoiser

The score function the ODE needs is given trivially by the denoiser:

$$\nabla_x \log p(x;\sigma) = \frac{D(x;\sigma) - x}{\sigma^2}$$

So in EDM you train a denoiser, not a score net, even though you sample using a probability-flow ODE that is normally derived in score-matching language. That equivalence is what makes the framework "elucidated" — every component is expressed through D.

---

## 3. Preconditioning (the "Precond" wrapper)

Training a network to output D directly is bad: the input x = y + n has variance σ² + σ_data², which ranges from ~σ_data² up to σ_max² ≈ 80² across noise levels. A vanilla network would have to handle a 160 000× variance swing. EDM solves this by training a *raw* network Fθ inside an analytic wrapper:

$$D_\theta(x;\sigma) = c_\text{skip}(\sigma)\,x + c_\text{out}(\sigma)\,F_\theta\!\big(c_\text{in}(\sigma)\,x;\;c_\text{noise}(\sigma)\big)$$

The four c-functions are derived (Appendix B.6 of EDM) by demanding:

- **cᵢₙ(σ)**: rescale input so Var = 1 → `1 / sqrt(σ² + σ_data²)`
- **cₒᵤₜ(σ)**: rescale output so the effective training target has Var = 1 → `σ · σ_data / sqrt(σ² + σ_data²)`
- **cₛₖᵢₚ(σ)**: choose the skip path so Fθ's amplification of its own error is minimised → `σ_data² / (σ² + σ_data²)`
- **cₙₒᵢₛₑ(σ)**: an arbitrary monotone embedding of σ for the timestep-conditioning MLP → `0.25 · ln(σ)`

What this buys you in practice:

- At **small σ** (clean image): cₛₖᵢₚ ≈ 1, cₒᵤₜ ≈ σ → Fθ is asked to predict only a small correction to the input.
- At **large σ** (pure noise): cₛₖᵢₚ ≈ 0, cₒᵤₜ ≈ σ_data → Fθ is asked to predict the *clean signal* directly, scaled to unit variance.
- Smooth interpolation between those regimes.

The effective per-σ loss weighting that comes out of putting all this into the L₂ loss is

$$\lambda(\sigma) = \frac{\sigma^2 + \sigma_\text{data}^2}{(\sigma\,\sigma_\text{data})^2}$$

which makes the *effective weight* in front of Fθ exactly 1 at every σ. So the network sees a constant-difficulty learning problem regardless of noise level — that's the whole point.

Karras also chose the **noise distribution at training** to be log-normal:

$$\ln\sigma \sim \mathcal{N}(P_\text{mean}, P_\text{std}^2)$$

with P_mean ≈ −1.2 and P_std ≈ 1.2. Putting more samples in the mid-σ band where the denoising task is hardest, and fewer at the trivial extremes, accelerates training noticeably.

---

## 4. Sampling

### 4.1 Karras σ schedule (the famous ρ=7 formula)

Discretising the ODE means picking a sequence σ₀ = σ_max > σ₁ > … > σ_{N−1} = σ_min, σ_N = 0. EDM uses

$$\sigma_i = \left(\sigma_\text{max}^{1/\rho} + \frac{i}{N-1}\big(\sigma_\text{min}^{1/\rho} - \sigma_\text{max}^{1/\rho}\big)\right)^{\rho}$$

ρ = 7 is empirical: it concentrates step density near σ_min where small absolute errors matter most, while still spending some steps at high noise. ρ = 3 equalises truncation error per step but performs worse for image quality, suggesting that errors near the data manifold dominate perceived quality.

### 4.2 Heun 2nd-order solver (Algorithm 1)

For each step i, evaluate dx/dt at tᵢ, take an Euler step to tᵢ₊₁, then re-evaluate dx/dt at the new point and apply the trapezoidal correction. That doubles the NFE per step but reduces the local truncation error from O(h²) to O(h³), which is a massive net win — full quality at 35 NFE rather than 250+. The final step (σ → 0) reverts to Euler to avoid a divide-by-zero.

### 4.3 Stochastic sampler (Algorithm 2, the "churn" sampler)

Sometimes a deterministic sampler accumulates errors that an ODE step alone can't recover from. EDM offers a controlled re-injection of noise:

- At each step, with probability/strength γᵢ, re-noise xᵢ from level tᵢ up to t̂ᵢ = (1 + γᵢ) tᵢ.
- Then Heun-step backward from t̂ᵢ to tᵢ₊₁.

The "noise injection then ODE step" alternation acts like a Langevin corrector that drags the state back toward the correct marginal at each level. Four hyperparameters control it: S_churn, S_tmin, S_tmax, S_noise. They have to be grid-searched per model.

---

## 5. The full EDM "design space" cheat-sheet

| Axis | EDM choice |
|---|---|
| ODE schedule σ(t) | t |
| Scaling s(t) | 1 |
| Discretisation {σᵢ} | Karras polynomial with ρ = 7 |
| Solver | Heun 2nd order (deterministic) or stochastic churn |
| Preconditioning | (cₛₖᵢₚ, cᵢₙ, cₒᵤₜ, cₙₒᵢₛₑ) above |
| Train σ distribution | ln σ ~ 𝒩(P_mean, P_std²) |
| Loss weighting | λ(σ) = (σ² + σ_data²) / (σ σ_data)² |
| Default σ_data | 0.5 (matches normalised image data) |
| Default σ_min, σ_max | 0.002, 80 |

These numerical defaults — particularly σ_data = 0.5, σ_min = 0.002, σ_max = 80, ρ = 7, P_mean = −1.2, P_std = 1.2 — are inherited verbatim by StormCast.

---

## 6. EDM inside StormCast

### 6.1 What StormCast is doing

StormCast emulates the HRRR convection-allowing model by predicting the next 1-hour atmospheric state from the previous one, conditioned on a coarser synoptic forecast. Following CorrDiff (Mardani et al.), it splits the conditional generative task into two stages:

**Phase 1 — Deterministic regression.** A U-Net Fξ regresses the conditional mean:

$$M_t = F_\xi(X_{t-1},\,S_t,\,I) \approx \mathbb{E}[X_t \mid X_{t-1}, S_t, I]$$

This learns the smooth, predictable, "ensemble-mean-ish" component of the next state. It blurs the precipitation field, but that's expected: at hourly cadence, the unblurred high-frequency content is genuinely stochastic.

**Phase 2 — EDM diffusion on residuals.** A second model Dψ (an `EDMPrecond` SongU-Net) learns the conditional distribution of the *residual*

$$R_t = X_t - M_t$$

given the conditioning bundle c = (X_{t−1}, S_t, M_t, I). In other words the diffusion model only has to model the unpredictable, high-frequency, multimodal part — exactly what generative modelling is good at. Sampling at inference reconstructs the full state as X̂_t = M_t + R̂_t.

This is why every sampler helper in [stormcast/utils/nn.py](stormcast/utils/nn.py) takes a `condition` tensor with the four conditioning fields concatenated on the channel dim.

### 6.2 Why EDM specifically

The StormCast paper cites Karras et al. directly (their reference [40] / [41]). Three reasons EDM is the right framework here:

1. **Independence of axes.** They could pick σ_data to match the residual statistics and pick a sampler step count to match operational latency, without breaking anything else. That's exactly the modularity Karras was selling.
2. **Heun + Karras schedule = few NFE.** Operational forecasting cares about wall-clock per ensemble member. EDM's deterministic sampler hits useful quality at ~18–35 steps; older DDPM-style schedules need hundreds.
3. **Stochastic sampler for free ensembling.** Each StormCast forecast member is generated by drawing a fresh latent N(0, σ_max² I) and running the sampler. Five members is "computationally cheap" precisely because per-sample NFE is small.

### 6.3 Concrete EDM hyperparameters used in this repo

From [CLAUDE.md](CLAUDE.md) and [stormcast/utils/nn.py:200-202](stormcast/utils/nn.py#L200-L202):

```
sigma_min  = 0.002
sigma_max  = 80
sigma_data = 0.5
rho        = 7
```

These are Karras' image-domain defaults transplanted unchanged. σ_data = 0.5 makes sense because the residuals R_t are *standardised* (per-channel z-scores) before training, so their per-channel std is ~1 and σ_data = 0.5 is a reasonable compromise — large enough that cₛₖᵢₚ is well-behaved at low σ, small enough that cₒᵤₜ scales sensibly at high σ.

The training-time σ distribution is the EDM log-normal. The architectural wrapper is `physicsnemo.models.diffusion.EDMPrecond` instantiated in [stormcast/utils/nn.py:48](stormcast/utils/nn.py#L48):

```python
EDMPrecond(
    img_resolution=img_resolution,
    img_channels=target_channels + conditional_channels,
    img_out_channels=target_channels,
    model_type="SongUNet",
    channel_mult=[1, 2, 2, 2, 2],
    ...
)
```

Note that `img_channels` includes both the noisy target channels *and* the conditioning channels; only the target channels come back out (`img_out_channels`). That is, conditioning is implemented by **channel-wise concatenation onto the noisy input** — the simplest possible conditioning scheme, which works because StormCast never has to align across spatial scales (the regression network already brought the synoptic field into the km-grid).

### 6.4 The diffusion training loss

For each sample (M_t, X_t, S_t, I) the trainer:

1. Computes the conditioning bundle and the residual target R_t = X_t − M_t.
2. Standardises R_t (per-channel z-score using the saved HighRes stats).
3. Draws σ ~ log-normal.
4. Adds noise: R̃ = R_t + n, n ~ 𝒩(0, σ²I).
5. Computes the EDM-preconditioned denoiser output D_ψ(R̃, σ; c).
6. Backprops the EDM loss

$$\mathcal{L} = \lambda(\sigma)\,\big\|D_\psi(R_t + n,\,\sigma;\,c) - R_t\big\|_2^2$$

with λ(σ) = (σ² + σ_data²) / (σ σ_data)².

The pretrained teacher checkpoint sitting in [exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus](exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus) is exactly that: 70 000 steps of this EDM loss.

### 6.5 The diffusion inference sampler

In [stormcast/utils/nn.py:174](stormcast/utils/nn.py#L174):

```python
def diffusion_model_forward(model, condition, shape, sampler_args={}):
    latents = torch.randn(*shape, device=condition.device, dtype=condition.dtype)
    return deterministic_sampler(
        model, latents=latents, img_lr=condition, **sampler_args
    )
```

The `deterministic_sampler` from `physicsnemo.utils.diffusion` is the EDM Heun sampler over the Karras ρ=7 schedule. It samples R̂ ~ p(R | c), and the calling code adds the regression mean back: X̂_t = M_t + σ_data · R̂ (the σ_data multiply un-standardises the residual prediction; see [stormcast/utils/nn.py:407](stormcast/utils/nn.py#L407)).

For ensembling, the only thing that varies between members is the initial `latents` draw. That cheap stochasticity is entirely owed to the diffusion stage — the regression stage is deterministic.

---

## 7. Why this matters for the distillation work in this repo

Every distillation method in [stormcast_distillation.md](stormcast_distillation.md) — Progressive, Consistency, ADD, DMD — is fundamentally a way of compressing the EDM teacher's *many-step Heun trajectory* into something that can be evaluated in 1–4 NFE. Each method respects EDM's framework but trades off different parts of it:

| Method | What it keeps from EDM | What it drops |
|---|---|---|
| **Progressive Distillation** | EDM σ schedule, EDM denoiser parameterisation, deterministic Heun teacher steps | Many NFE; halves the schedule per phase |
| **Consistency Distillation** | EDM noise distribution, σ_data preconditioning, Heun teacher operator Φ | The whole ODE trajectory: replaced with a self-consistent "any σ → σ_min" map |
| **Adversarial Distillation (ADD)** | Teacher denoiser as a stop-grad regulariser; EDM sampling at inference (1 step) | Per-σ MSE loss; replaced by hinge GAN + spectral regulariser |
| **DMD** | EDM-preconditioned student; σ_max-only forward pass | Score matching; replaced by distribution matching |

Three EDM facts have *direct* operational consequences for distillation in this codebase:

- **Copy σ_min, σ_max, σ_data, ρ from the teacher.** Distillation losses depend on the σ schedule the teacher was trained against; mismatches silently break alignment of the student with the teacher's score field. The CLAUDE wiki flags this as one of the easiest ways to break a run.
- **The teacher Φ-step in CD must be Heun, not Euler.** The teacher was trained to be sampled with a Heun trajectory; reusing Euler in the consistency target makes the student emulate the *wrong* mapping. This is encoded in [physicsnemo/experimental/metrics/diffusion/consistency_loss.py](physicsnemo/experimental/metrics/diffusion/consistency_loss.py).
- **σ_data = 0.5 + per-channel z-score normalisation is the trick that lets EDM apply unchanged to atmospheric data.** Once you have channel-wise unit-variance residuals, the image-domain Karras defaults work out of the box. The Taiwan-domain residuals satisfy this approximately for t2m/u10/v10 but **qpepre is heavy-tailed (mean 0.18, std 9.12)** and violates the unit-variance assumption that motivated cₒᵤₜ. That's the deep reason every distillation method in this repo needs a special term for qpepre (Pseudo-Huber, asinh transform, spectral L1) — they're all patches around the fact that EDM's loss-weighting derivation no longer holds for that one channel.

---

## 8. Summary

EDM is a denoising-diffusion design framework that:

- expresses the problem as learning an L₂-optimal denoiser D(x; σ) instead of a score,
- wraps that denoiser in an analytic preconditioner (cₛₖᵢₚ, cᵢₙ, cₒᵤₜ, cₙₒᵢₛₑ) that keeps the raw network's training problem at constant difficulty across σ,
- samples with a Heun 2nd-order solver on a Karras ρ=7 schedule, optionally with stochastic churn,
- and decouples training, sampling, and preconditioning so each can be tuned independently.

StormCast uses EDM as a residual generative model on top of a deterministic regression backbone. The teacher checkpoint is an `EDMPrecond` SongU-Net trained with the standard Karras hyperparameters (σ_min=0.002, σ_max=80, σ_data=0.5, ρ=7) on per-channel-standardised residuals R_t = X_t − M_t. Inference is the Heun deterministic sampler; ensemble spread comes from the latent noise draw. All distillation work in this repository starts from this teacher and aims to compress its multi-step Heun trajectory into a 1-to-few-step student while preserving its sampling distribution — particularly for the heavy-tailed qpepre channel where EDM's unit-variance assumptions are violated and special losses are needed.
