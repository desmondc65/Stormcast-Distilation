# Distillation Methods for a Taiwan-Region RWRF StormCast

Your setup is a **regional, smaller-scale** StormCast than NVIDIA's: Taiwan domain (~21.6°–25.6°N, 119.75°–122.25°E), 224×128 grid, 1-hour steps, with the teacher's generative target being a **4-channel RWRF high-resolution state** (t2m, u10, v10, qpepre) conditioned on a 24-channel low-resolution synoptic state + 2 invariants. This changes several design decisions vs. the 99-channel NOAA HRRR case — in ways that make distillation *easier*, but also put **qpepre (precipitation)** front and center as the hardest variable to distill well.

---

## 0. Your Data & Setup

### Arrays

| Array | Shape | Channels |
|---|---|---|
| LowRes (conditioning) | (T, 24, 224, 128) | mslp, t2m, u10, v10; q/t/u/v/z at 1000/850/500/250 hPa |
| HighRes (target) | (T, 4, 224, 128) | **t2m, u10, v10, qpepre** |
| Invariants | (2, 224, 128) | lsm (land-sea mask), orog (orography) |

- Training: 21,216 hourly steps (~2.5 years, 2019/08 – 2021/12)
- Validation: 8,760 hourly steps (2022 full year)
- Normalization stats are provided per-channel for both LowRes (24) and HighRes (4)

### Your StormCast variables (renamed to match your data)

| Symbol | Meaning |
|---|---|
| $X_t \in \mathbb{R}^{4 \times 224 \times 128}$ | **RWRF HighRes** state at hour $t$: (t2m, u10, v10, qpepre) |
| $X_{t-1}$ | Previous RWRF HighRes state |
| $S_t \in \mathbb{R}^{24 \times 224 \times 128}$ | **LowRes** synoptic conditioning at hour $t$ |
| $I \in \mathbb{R}^{2 \times 224 \times 128}$ | Invariants (lsm, orog) — constant across time |
| $\xi$ | Regression-network parameters (frozen) |
| $F_\xi$ | Regression network |
| $M_t = F_\xi(X_{t-1}, S_t, I)$ | Deterministic mean forecast (4 channels) |
| $R_t = X_t - M_t$ | **Residual** — what the diffusion generates |
| $c = (X_{t-1}, S_t, M_t, I)$ | Full conditioning bundle |
| $\sigma$ | EDM noise level |
| $\sigma_{\text{data}}$ | EDM data-scale hyperparameter |
| $\epsilon \sim \mathcal{N}(0, I)$ | Noise |
| $z_\sigma = R_t + \sigma \epsilon$ | Noised residual |
| $\psi$ | Teacher StormCast diffusion parameters (frozen) |
| $D_\psi(z_\sigma, \sigma; c)$ | Teacher denoiser, Karras preconditioned |
| $w(\sigma) = (\sigma^2 + \sigma_{\text{data}}^2)/(\sigma \cdot \sigma_{\text{data}})^2$ | EDM weighting |
| $\beta_k$, $k \in \{1,2,3,4\}$ | Per-variable weights for (t2m, u10, v10, qpepre) |

### What's different from the 99-channel case

| Aspect | NVIDIA StormCast | Your RWRF StormCast |
|---|---|---|
| Target channels | 99 | **4** |
| Domain | CONUS (3km) | Taiwan (~28 km north–south, 16 km east–west per tile?) |
| Key hard variable | Radar reflectivity | **qpepre (precipitation)** |
| Ensemble task | Critical | Still useful, but single-sample skill may matter more for your thesis |
| Spectral concern | Multi-variable | Mostly about **precipitation spatial structure** |

**Two things about your data drive every design below:**

1. **qpepre is the hard channel.** Looking at your HighRes stats (mean 0.18, std 9.12), precipitation is extremely heavy-tailed, sparse, and non-Gaussian — the classic failure mode for MSE distillation. Every method below needs special handling for this channel.

2. **Only 4 output channels.** This is much easier than 99 channels for distillation — the loss landscape is simpler, and you can afford per-channel custom loss terms without it becoming a bookkeeping nightmare.

---

## 1. Progressive Distillation (PD)

**Safest first attempt.** Inherits teacher behavior by construction; just compresses the ODE trajectory.

### Method-specific variables

| Symbol | Meaning |
|---|---|
| $\eta$ | Student parameters (init from teacher each round) |
| $D_\eta$ | Student denoiser |
| $N$ | Current student step count |
| $\sigma, \sigma'$ | Consecutive noise levels on student's grid |
| $\tilde{R}$ | Target: DDIM back-out from two teacher **Euler** PF-ODE steps $(z_\sigma, \sigma) \to \sigma'$ (Salimans & Ho 2022 Alg. 2) |
| $\hat{R}_\eta = D_\eta(z_\sigma, \sigma; c)$ | Student's denoiser output at $\sigma$ (compared in denoised-prediction space, not trajectory space) |

### Setup

- Teacher: your pretrained RWRF StormCast, frozen.
- Student $D_\eta$: same U-Net architecture, weights copied from teacher.
- Step schedule: if your inference uses ~18 Heun steps, start at $N=16$ and halve: $16 \to 8 \to 4 \to 2$.

### Training step

For a sample $(X_{t-1}, S_t, X_t, I)$ from your zarr train split:
1. Run frozen $F_\xi$ to get $M_t$; form $R_t = X_t - M_t$.
2. Sample $\sigma$ from student grid; draw $\epsilon$; form $z_\sigma = R_t + \sigma \epsilon$.
3. Run **two teacher Euler PF-ODE steps** $(z_\sigma, \sigma) \to \sigma_{\text{mid}} \to \sigma'$, then DDIM-invert to an implied denoised target $\tilde{R}$.
4. Student predicts $\hat{R}_\eta$ directly (denoiser output at $\sigma$), and the loss is MSE in denoised-prediction space.

### Loss

$$
\mathcal{L}_{\text{PD}}^{\text{RWRF}} = \mathbb{E}\!\left[ w(\sigma) \sum_{k=1}^{4} \beta_k \, \| \hat{R}_\eta^{(k)} - \tilde{R}^{(k)} \|_2^2 \right]
$$

### Per-variable weights $\beta_k$ — concrete recommendation

Since you normalize with per-channel stds, the simplest starting point is $\beta_k = 1$ on z-scored residuals. But for qpepre specifically:

- **Option**: up-weight qpepre (e.g., $\beta_4 = 2$ or $3$) since it's the variable most vulnerable to PD's mode-averaging.
- **Better option**: apply a log or asinh transform to qpepre *before* computing residuals, so that the diffusion target is smoother. If your teacher was trained this way, keep the same transform.

### Watch-outs for your setup

- **Precipitation diversity collapse.** MSE is mode-averaging; after a couple of PD rounds, qpepre tends toward smooth, low-amplitude "climatological drizzle." Track the wet/dry pixel distribution and 95th/99th percentiles of qpepre each round.
- **Spectral blurring of qpepre.** Add an auxiliary log-PSD matching term on the precipitation channel specifically.
- Expect 4-step usable, 2-step risky, 1-step probably unusable for qpepre without further tricks.

---

## 2. Consistency Distillation (CD)

**Best quality/effort tradeoff for your setup.** Multi-step sampling mode is native, noise injection between steps preserves ensemble spread.

### Method-specific variables

| Symbol | Meaning |
|---|---|
| $\theta$ | Student (online) parameters |
| $\theta^-$ | Target parameters (EMA of $\theta$) |
| $\mu$ | EMA decay |
| $F_\theta(z, \sigma; c)$ | Raw network output |
| $f_\theta(z, \sigma; c)$ | Consistency function |
| $N$ | Number of discretization points |
| $\sigma_1 < \dots < \sigma_N$ | EDM $\rho=7$ noise grid |
| $n \sim \mathcal{U}\{1, \dots, N-1\}$ | Random index |
| $\Phi(z, \sigma; c)$ | One Heun step of teacher PF-ODE |
| $\hat{z}_{\sigma_n}^\Phi$ | Trajectory estimate at $\sigma_n$ |
| $\lambda(\sigma_n)$ | Per-step weight, $1/(\sigma_{n+1} - \sigma_n)$ |
| $d_{\text{atmos}}$ | Atmospheric-field distance |

### Parameterization (Karras preconditioning, same as teacher)

$$
f_\theta(z, \sigma; c) = c_{\text{skip}}(\sigma) \, z + c_{\text{out}}(\sigma) \, F_\theta(z, \sigma; c)
$$

### Teacher one-step update (Heun — match your sampler)

$$
\hat{z}_{\sigma_n}^\Phi = z_{\sigma_{n+1}} + (\sigma_n - \sigma_{n+1}) \, \Phi(z_{\sigma_{n+1}}, \sigma_{n+1}; c)
$$

### Training step

1. Draw $(X_{t-1}, S_t, X_t, I)$; compute $M_t$, $R_t$.
2. Sample $n$; draw $\epsilon$; form $z_{\sigma_{n+1}} = R_t + \sigma_{n+1} \epsilon$.
3. Teacher Heun step → $\hat{z}_{\sigma_n}^\Phi$.
4. Compute $f_\theta$ at $\sigma_{n+1}$ and $f_{\theta^-}$ at $\sigma_n$; minimize distance.
5. EMA: $\theta^- \leftarrow \mu \theta^- + (1-\mu)\theta$.

### Loss

$$
\mathcal{L}_{\text{CD}}^{\text{RWRF}} = \mathbb{E}\!\left[ \lambda(\sigma_n) \cdot d_{\text{atmos}}\!\left( f_\theta(z_{\sigma_{n+1}}, \sigma_{n+1}; c), \; f_{\theta^-}(\hat{z}_{\sigma_n}^\Phi, \sigma_n; c) \right) \right]
$$

### Choosing $d_{\text{atmos}}$ for 4-channel RWRF

> **Status:** This hybrid distance is already wired into `ConsistencyDistillationLoss` via the `channel_weights`, `spectral_channels`, and `spectral_weight` parameters — see [`physicsnemo/experimental/metrics/diffusion/consistency_loss.py`](physicsnemo/experimental/metrics/diffusion/consistency_loss.py) and [`stormcast/config/training/consistency.yaml`](stormcast/config/training/consistency.yaml). The formulas below describe what the code computes, not a future change.

With only 4 channels, you can afford a **hybrid loss** with explicit per-channel terms:

$$
d_{\text{atmos}}(x, y) = \underbrace{\sum_{k=1}^{3} \beta_k \, d_{\text{Huber}}(x_k, y_k)}_{\text{t2m, u10, v10}} + \underbrace{\beta_4 \, d_{\text{Huber}}(x_4, y_4) + \alpha_{\text{spec}} \, \| \log\text{PSD}(x_4) - \log\text{PSD}(y_4) \|_1}_{\text{qpepre + spectral}}
$$

where Pseudo-Huber is $d_{\text{Huber}}(a, b) = \sqrt{\|a-b\|_2^2 + c^2} - c$.

- Pseudo-Huber on all channels (improved CT recommends this; robust to qpepre outliers).
- Extra log-PSD term on **qpepre only** preserves precipitation spatial structure.
- Everything computed on z-scored tensors using your per-channel HighRes stats (means/stds you already have).

### Why CD fits your thesis well

- **Operational knob:** expose `num_steps` for the reviewer. 2-step CD often matches teacher quality at 18 steps.
- **Single training run** — no progressive rounds, saving compute.
- **Preserves stochasticity** via noise injection between steps — important if you want to report ensemble metrics on the 2022 validation year.

### Watch-outs

- Growing-$N$ schedule (improved CT) is worth copying for stability.
- Heun (not Euler) for the teacher step.

---

## 3. Adversarial Diffusion Distillation (ADD)

Highest ceiling, highest risk. Works well here **only if** you handle the discriminator backbone carefully — no pretrained perceptual model exists for Taiwan-region RWRF fields.

### Method-specific variables

| Symbol | Meaning |
|---|---|
| $\theta$ | Student generator parameters |
| $\varphi$ | Discriminator head parameters |
| $\{\tau_1, \dots, \tau_4\}$ | Student's small set of EDM levels |
| $\sigma_s, \sigma_t$ | Student and teacher levels, $\sigma_t > \sigma_s$ |
| $\epsilon, \epsilon'$ | Independent noises |
| $z_{\sigma_s} = R_t^{\text{real}} + \sigma_s \epsilon$ | Noised real residual |
| $\hat{R}_\theta$ | Student one-step residual estimate |
| $\hat{z}_{\sigma_t} = \hat{R}_\theta + \sigma_t \epsilon'$ | Re-noised for teacher |
| $\hat{R}_\psi = D_\psi(\hat{z}_{\sigma_t}, \sigma_t; c)$ | Teacher target, stop-grad |
| $D_\varphi(R, c)$ | Conditional discriminator |
| $R_1$ | Gradient penalty on reals |
| $\gamma, \lambda_1, \lambda_2$ | Weights |

### Discriminator backbone — your best option

Since you have no pretrained atmospheric backbone and your domain is small (224×128, 4 channels), I'd recommend:

**Reuse teacher U-Net encoder features.** Take the teacher's encoder, freeze it, attach small trainable conv heads at multiple depths. Condition on $c$ by concatenating $(M_t, I, \text{upsampled}(S_t))$ as extra channels. Cheap, stable, and the features are already tuned to your exact domain.

### Training step

1. Real $(X_{t-1}, S_t, X_t, I)$; $R_t^{\text{real}} = X_t - M_t$.
2. Sample $\sigma_s \in \{\tau_i\}$; $z_{\sigma_s} = R_t^{\text{real}} + \sigma_s \epsilon$.
3. Student: $\hat{R}_\theta = \hat{R}_\theta(z_{\sigma_s}, \sigma_s; c)$.
4. $\sigma_t > \sigma_s$; re-noise; teacher produces $\hat{R}_\psi$ (stop-grad).

### Losses

**Generator adversarial (hinge):**
$$
\mathcal{L}_{\text{adv}}^{G}(\theta) = -\mathbb{E}\!\left[ D_\varphi(\hat{R}_\theta, c) \right]
$$

**Discriminator (hinge + $R_1$):**
$$
\mathcal{L}_{\text{adv}}^{D}(\varphi) = \mathbb{E}\!\left[\max(0, 1 - D_\varphi(R_t^{\text{real}}, c))\right] + \mathbb{E}\!\left[\max(0, 1 + D_\varphi(\hat{R}_\theta, c))\right] + \gamma R_1
$$

**Distillation:**
$$
\mathcal{L}_{\text{distill}}(\theta) = \mathbb{E}\!\left[ w(\sigma_t) \sum_{k=1}^{4} \beta_k \, \| \hat{R}_\theta^{(k)} - \operatorname{sg}[\hat{R}_\psi^{(k)}] \|_2^2 \right]
$$

**Spectral regularizer on qpepre** (essential for your setup):
$$
\mathcal{L}_{\text{spec}}(\theta) = \alpha_{\text{spec}} \, \| \log\text{PSD}(\hat{R}_\theta^{(4)}) - \log\text{PSD}(R_t^{\text{real},(4)}) \|_1
$$

**Total:**
$$
\mathcal{L}_{\text{ADD}}^{\text{RWRF}}(\theta) = \mathcal{L}_{\text{adv}}^{G} + \lambda_1 \mathcal{L}_{\text{distill}} + \lambda_2 \mathcal{L}_{\text{spec}}
$$

### Watch-outs

- **Precipitation mode collapse is the #1 risk.** A weak discriminator on a sparse, heavy-tailed channel will let the generator produce "always slightly wet" fields that fool it. Monitor wet/dry pixel fraction and tail quantiles.
- **Small domain helps** (224×128 is manageable for stable GAN training) but per-channel skill is easier to compromise.
- **Autoregressive drift** — evaluate 6h and 12h rollouts throughout training.

---

## 4. Cross-Method Quick Reference

| Role | PD | CD | ADD |
|---|---|---|---|
| Student params | $\eta$ | $\theta$ | $\theta$ |
| Teacher params | $\psi$ | $\psi$ (via $\Phi$) | $\psi$ |
| Auxiliary params | — | $\theta^-$ (EMA) | $\varphi$ (discriminator) |
| Target signal | $\tilde{R}$ (2 teacher Heun steps) | $f_{\theta^-}(\hat{z}_{\sigma_n}^\Phi, \sigma_n; c)$ | Real $R_t^{\text{real}}$ + teacher $\hat{R}_\psi$ |
| Student level | $\sigma$ | $\sigma_{n+1}$ | $\sigma_s$ |
| Target level | $\sigma'$ | $\sigma_n$ | $\sigma_t$ |
| Rounds | $\log_2 N$ | 1 | 1 |
| Real data needed? | No | No | **Yes** |
| qpepre difficulty | Hardest (mode collapse) | Manageable with Huber+PSD | Risky (GAN on sparse data) |

---

## 5. Recommended Thesis Path

Given your thesis scope and the 4-channel/small-domain nature of your setup:

1. **Baseline: PD to 4 steps.** ~4× speedup, low risk. Report RMSE on t2m/u10/v10 and CSI/FSS on qpepre at common thresholds (0.1, 1, 10 mm/h).

2. **Main contribution: CD with Huber + qpepre log-PSD loss.** Train once, expose `num_steps` ∈ {1, 2, 4}. Show the quality/speed tradeoff curve. 2-step CD matching teacher-at-18-steps is a strong result.

3. **Optional: ADD** as an "if time permits" extension, using the teacher encoder as frozen discriminator backbone. Frame as exploratory.

### Metrics suite (use your 2022 validation year)

- **Deterministic**: per-channel RMSE vs. RWRF target (use your saved stats for de-normalization).
- **Precipitation-specific**: CSI, FSS, bias, frequency-of-exceedance at 0.1 / 1 / 5 / 10 / 20 mm/h. These distinguish "looks okay on average" from "actually captures rain events."
- **Spectral**: radially averaged log-PSD of qpepre, compared teacher vs. student.
- **Rollout**: 1h, 3h, 6h, 12h RMSE/CSI — catches autoregressive drift, which is the real failure mode.
- **Ensemble (optional)**: 10-member CRPS and rank histograms for qpepre if you run ensembles.

### A few concrete next steps to decide on

- Does your teacher apply a transform (log / asinh) to qpepre before the diffusion? If yes, match it in the student. If no, strongly consider adding one before distillation — it makes every method above more stable.
- What EDM parameters ($\sigma_{\min}, \sigma_{\max}, \sigma_{\text{data}}, \rho$) did your teacher train with? Copy them exactly into the student.
- How many Heun steps does your current teacher inference use? That sets $N$ for PD and the target for CD to match.

If you can share the teacher's training config or loss/prediction head details, I can tighten the loss specs (especially around qpepre transform and EDM hyperparameters) further.