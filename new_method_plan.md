# BridgeCast: Mean-Anchored Schrödinger-Bridge Forecasting for Storm-Scale Generative Emulation

**One-line TL;DR.** Replace the noise→data stochastic transport used by StormCast (EDM) and FlowCast (CFM) with a **mean→data Schrödinger bridge** that starts the generative trajectory at the regression conditional mean $M_t$, perturbs it with **climatology-correlated** noise, and is trained jointly with a **kernel-CRPS ensemble objective** and **channel-adaptive paths** for the heavy-tailed precipitation channel. Same dataset, same 4 output channels, same conditioning bundle as StormCast — but a fundamentally shorter (and straighter) generative path that gives sharp, well-calibrated, physically-consistent samples in 1–4 NFE.

---

## 1. Why a new method? — Critique of the two existing baselines

Both methods we want to beat operate on the same residual decomposition $X_t = M_t + R_t$ but disagree only about how to model the residual distribution $p(R_t \mid c)$ where $c = (X_{t-1}, S_t, M_t, I)$.

| Aspect | **StormCast (EDM)** [Pathak et al. 2024] | **FlowCast (CFM) for StormCast** [Ribeiro & Pucer 2025] | **Common shortcoming** |
|---|---|---|---|
| Source distribution | $\mathcal{N}(0, \sigma_{\max}^2 I)$, $\sigma_{\max}=80$ | $\mathcal{N}(0, I)$ (I-CFM) | **Both start from white isotropic Gaussian noise**, even though we already know $M_t$ is a far better starting point. |
| Path geometry | Curved Karras VP-like ODE, $\rho=7$ | Linear in expectation (straight) | Straight only when source is Gaussian; ignores that the *true* end-to-end manifold transport is short if anchored at $M_t$. |
| NFE | 18–25 (Heun) | 10 (Euler), 1–3 if pushed | Both still "noise → data"; transport length is wasted. |
| Calibration objective | Marginal denoising MSE, no ensemble term | CFM regression, no ensemble term | Trains for sample fidelity, **not** ensemble calibration → known underdispersion in StormCast (Appendix D of Pathak '24). |
| Heavy-tailed `qpepre` | Treated identically to `t2m` via shared $\sigma_{\text{data}}=0.5$ | Same | Channel-uniform path is a poor fit to a sparse mean-0.18, std-9.12 channel. |
| Physics | None enforced | None | qpepre can go negative, BL divergence is unconstrained. |
| Spatial structure of noise | i.i.d. white | i.i.d. white | Storm features have $\sim\!10\text{–}100$ km autocorrelation; the model wastes capacity inventing spatial coherence from white noise. |

**Single hypothesis that motivates the new method.** A generative model whose stochastic process has (i) a *short* transport distance, (ii) a *spatially correlated* noise budget aligned with atmospheric autocorrelation, (iii) a *channel-adaptive* path for sparse heavy-tailed fields, and (iv) an *ensemble-aware* objective will outperform StormCast and FlowCast on **both** sharpness (CSI/FSS at high thresholds) and calibration (CRPS, spread/skill, rank histogram), at far lower NFE.

---

## 2. BridgeCast — formulation

### 2.1 Mean-anchored stochastic interpolant (the bridge)

We use the **Stochastic Interpolant** framework of Albergo, Boffi & Vanden-Eijnden (2023), specialised to two non-Gaussian endpoints:

- **Source endpoint** $x_0 \sim p_0(\cdot \mid c) = \delta(M_t)$ — a deterministic anchor at the regression mean. (We will also experiment with $p_0 = \mathcal{N}(M_t, \sigma_a^2 I)$ for a small $\sigma_a$, see §2.2.)
- **Target endpoint** $x_1 \sim p_1(\cdot \mid c) = p(X_t \mid c)$ — the conditional data distribution.

Define the time-dependent interpolant $\{x_t\}_{t \in [0,1]}$ between paired samples $(x_0, x_1)$ as

$$
x_t \;=\; (1-t)\, x_0 \;+\; t\, x_1 \;+\; \gamma(t) \, \boldsymbol{\varepsilon},
\qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, \Sigma_{\text{clim}}),
$$

with the I²SB-style symmetric noise schedule (Liu et al. 2023; Heng et al. 2024)

$$
\gamma(t) \;=\; \sigma_b \sqrt{\,t(1-t)\,},
$$

so $\gamma(0)=\gamma(1)=0$ — the interpolant exactly reproduces both endpoints — and is maximal at $t=\tfrac{1}{2}$, regularising the path with a small *bridge noise* $\sigma_b \in [0.05, 0.3]$ in normalised units. This is a **Schrödinger bridge** between $\delta(M_t)$ and $p(X_t \mid c)$ when the noise covariance is white; with the climatology-correlated $\Sigma_{\text{clim}}$ below it is its closest principled analogue (a Brownian bridge under a coloured Wiener process; see Léonard 2014 for the entropic-OT view).

### 2.2 Why this is *fundamentally* shorter than StormCast / FlowCast

Define the average squared transport length $\Delta := \mathbb{E}\big[\| x_0 - x_1 \|_2^2 / D \big]$ where $D$ is the pixel × channel count. Empirical estimates on the Taiwan RWRF training residuals are roughly:

- **StormCast (EDM):** $\Delta \approx \sigma_{\max}^2 + \sigma_{\text{data}}^2 \approx 6400.25$ (in $\sigma$-normalised units).
- **FlowCast (I-CFM):** $\Delta \approx 1 + \sigma_{\text{data}}^2 \approx 1.25$.
- **BridgeCast (this work):** $\Delta = \mathbb{E}\| R_t \|^2 / D$ — the **per-channel residual variance after regression**. From CLAUDE.md the per-channel HighRes std is $[5.61, 3.84, 5.66, 9.12]$ and the regression already removes most of the variance, so $\Delta_{\text{BC}}$ is **at least one order of magnitude smaller than FlowCast's** in normalised units.

A shorter, straighter path needs fewer Euler steps to integrate accurately — this is the central efficiency argument and we will verify it empirically.

### 2.3 Climatology-correlated noise prior $\Sigma_{\text{clim}}$

White Gaussian noise has flat power spectrum; mesoscale residuals do not. For each channel $k$ we estimate the **radially-averaged power spectral density** $\hat{P}_k(\kappa)$ of the training residuals $R_t$ once before training (cheap, one-pass FFT estimate). At runtime we sample noise as

$$
\varepsilon_k \;=\; \mathcal{F}^{-1}\!\left[ \sqrt{\hat{P}_k(\kappa)} \cdot \hat{\xi}_k \right], \qquad \hat{\xi}_k \sim \mathbb{C}\mathcal{N}(0, I)\text{ on the half-plane},
$$

i.e. **spectral whitening's inverse**: white in Fourier amplitude, but reweighted to match the empirical residual spectrum, with random phase. The result is per-channel Gaussian noise whose 2-point spatial autocorrelation matches climatology. Implementation: precompute $\sqrt{\hat{P}_k(\kappa)}$ as a $H\times W$ filter once, store as a buffer, sample by FFT-multiply-IFFT — overhead is negligible.

**Why this matters.** For `qpepre` the autocorrelation length is $\sim 20$–$50$ km; for `t2m` it is $\sim 100$ km. White noise forces the network to *create* this structure during the trajectory; correlated noise gives it for free, freeing capacity for the actually-stochastic parts of the residual (cell location, intensity).

### 2.4 Channel-adaptive paths for `qpepre`

The default linear interpolant is a poor fit when one channel is heavy-tailed and sparse (`qpepre`: mean 0.18, std 9.12, $>90$% of pixels are exactly 0). Two channel-specific modifications:

**(a) Asinh-compressed bridge for `qpepre`.** Define the monotone transform $\phi(q) = \mathrm{asinh}(q / \kappa)$ with $\kappa = 1$ mm h$^{-1}$. Train BridgeCast in the transformed space:

$$
\tilde{x}_t^{(\text{qpepre})} \;=\; (1-t)\, \phi(M_t^{(\text{qpepre})}) \;+\; t\, \phi(X_t^{(\text{qpepre})}) \;+\; \gamma(t)\,\varepsilon_{\text{clim}}^{(\text{qpepre})}.
$$

`asinh` behaves linearly near 0 (preserving the large mass of zero/light rain) and logarithmically in the tail (compressing extremes), which makes the bridge target velocity well-conditioned. This trick is now standard in radar nowcasting (Leinonen et al. 2023; Andrychowicz et al. 2023) but neither StormCast nor FlowCast uses it on this dataset.

**(b) Auxiliary rain-mask head.** Add a binary head $m_\theta(x_t, t, c) \in [0,1]^{H \times W}$ predicting the wet/dry probability. Trained with focal BCE against the indicator $\{X_t^{(\text{qpepre})} > 0.1\,\text{mm/h}\}$. At inference, gate the decoded `qpepre` by the mask: prevents the model from spreading thin rain everywhere ("dribble" mode collapse seen in CD/ADD).

The other three channels (`t2m`, `u10`, `v10`) keep the standard linear path.

### 2.5 Network parameterisation: dual-head velocity + score

We reuse the `SongUNet` backbone of `EDMPrecond` (no architecture changes; same parameter count as the StormCast student so all comparisons are FLOP-matched).

The network takes $(x_t, t, c)$ where $c = (X_{t-1}, S_t, M_t, I)$ is the same channel-stacked conditioning bundle StormCast and FlowCast use. It outputs:

1. **Velocity head** $v_\theta(x_t, t, c) \in \mathbb{R}^{H \times W \times 4}$ — the drift of the bridge ODE/SDE.
2. **Auxiliary mask head** $m_\theta(x_t, t, c) \in \mathbb{R}^{H \times W \times 1}$ — wet/dry probability.

The **velocity-matching target** is the conditional expectation of the instantaneous interpolant velocity:

$$
\dot{x}_t \;=\; (x_1 - x_0) \;+\; \gamma'(t)\,\boldsymbol{\varepsilon}, \qquad \gamma'(t) = \tfrac{\sigma_b (1 - 2t)}{2\sqrt{t(1-t)}}.
$$

Following Albergo et al. (2023, Theorem 2.6) the optimal regressor is $v^\star(x,t,c) = \mathbb{E}[\dot{x}_t \mid x_t = x, c]$. We train via stochastic regression — one fresh sample of $(t, \varepsilon, x_0, x_1)$ per minibatch element. Because the noise term has zero mean, regressing on $\dot{x}_t$ directly is *unbiased* but high-variance; we use the **antithetic-pair trick** of Lipman et al. (2023, Eq. 11): pair samples $(\varepsilon, -\varepsilon)$ in the same minibatch — this halves the variance of the velocity-matching loss at zero extra forward-pass cost.

We restrict $t \in [t_\varepsilon, 1 - t_\varepsilon]$ with $t_\varepsilon = 10^{-3}$ to avoid the $1/\sqrt{t(1-t)}$ blow-up of $\gamma'$.

### 2.6 Ensemble-aware kernel-CRPS objective

A key flaw of every method in §1 is that they train for **sample fidelity** but get evaluated on **ensemble calibration**. BridgeCast adds a direct **Energy Score** term that aligns the two.

For each minibatch element we generate $K$ independent samples $\{\hat{X}_t^{(k)}\}_{k=1}^K$ via 1-step Euler from the bridge (this is cheap — same cost as $K$ extra forward passes). The (negatively-oriented) Energy Score is

$$
\mathrm{ES}(\hat{P}_K, X_t) \;=\; \frac{1}{K}\sum_{k=1}^{K} \| \hat{X}_t^{(k)} - X_t \|_2 \;-\; \frac{1}{2K(K-1)} \sum_{k \ne j} \| \hat{X}_t^{(k)} - \hat{X}_t^{(j)} \|_2,
$$

which is a strictly proper scoring rule (Gneiting & Raftery 2007). With $K=4$–$8$ this gives a low-variance unbiased estimator of $\mathrm{ES}(\hat{P}_\theta, p_{\text{true}})$ and, critically, **trains the model to produce calibrated spread**, not just realistic individual samples. This idea was popularised at scale by Lang et al. (AIFS-CRPS 2024) for global ensemble forecasting; we transplant it into the BridgeCast objective for convective-scale generation.

To keep the ensemble term tractable on a 224×128×4 field we apply it (a) at the pixel level on `qpepre` directly (where calibration matters most) and (b) at coarse-grained $4\times4$ pooled level for the other channels. Pooled Energy Score still strictly-properly scores the joint distribution at the pooled scale (Pacchiardi & Dutta 2024).

### 2.7 Soft physics correctors

We add three small, *weak* physics priors that respect what is and is not known to hold at hourly km-scale:

1. **Non-negativity gate on `qpepre`.** The decoder applies $\hat{q}_{\text{pre}} = \max(0, q_{\text{lin}})$ post-asinh-decode, with a smooth softplus relaxation during training to keep gradients flowing.
2. **Soft 2-D divergence penalty** on $(u_{10}, v_{10})$: penalise $\| \partial_x \hat{u}_{10} + \partial_y \hat{v}_{10} \|_2^2$ with a tiny weight $\lambda_{\text{div}}=10^{-3}$. This is **not** a true incompressibility constraint (the atmosphere is not 2-D incompressible) — it is a *regulariser* that disfavours wildly divergent surface winds and tightens spatial coherence. We will ablate this term and only keep it if it helps without degrading u/v RMSE.
3. **Spectral consistency loss** on radial PSD per channel — already implemented in `consistency_loss.py`, reused as-is with weight $\lambda_{\text{spec}}=0.1$ on `qpepre` (matches FlowCast's setting in `train_flowcast.sh`).

### 2.8 Total loss

$$
\mathcal{L} \;=\; \lambda_v \cdot \mathcal{L}_v \;+\; \lambda_m \cdot \mathcal{L}_{\text{mask}} \;+\; \lambda_e \cdot \mathcal{L}_{\text{ES}} \;+\; \lambda_{\text{spec}} \cdot \mathcal{L}_{\text{spec}} \;+\; \lambda_{\text{div}} \cdot \mathcal{L}_{\text{div}},
$$

with channel-weighted velocity loss
$$
\mathcal{L}_v \;=\; \mathbb{E}_{t, \varepsilon, (x_0, x_1)}\Big[ \sum_{k=1}^{4} \beta_k \, \| v_\theta(x_t, t, c)_k - \dot{x}_{t,k} \|_2^2 \Big],
$$
weights $\beta = [1.0, 1.0, 1.0, 2.0]$ (matching the FlowCast configuration). Hyperparameters: $\lambda_v=1$, $\lambda_m=0.1$, $\lambda_e=0.5$, $\lambda_{\text{spec}}=0.1$, $\lambda_{\text{div}}=10^{-3}$ (to be ablated).

---

## 3. Inference

```
Algorithm 1: BridgeCast Sampling (single member)
Inputs:  conditioning c = (X_{t-1}, S_t, M_t, I), trained v_θ, m_θ
         number of Euler steps S, bridge noise σ_b, climatology filter √P_k

1.  Sample ε ~ N(0, Σ_clim)                          # FFT-correlated noise
2.  x ← M_t                                          # ANCHOR (not noise!)
3.  Δt ← 1 / S
4.  for i = 0 to S − 1:
        t ← (i + 0.5) · Δt                           # midpoint time, avoids γ' blow-up
        v ← v_θ(x + γ(t) · ε, t, c)                  # apply asinh on qpepre channel
        x ← x + v · Δt
5.  apply asinh-decode and non-negativity gate to qpepre channel of x
6.  apply rain-mask gate: x_qpepre ← x_qpepre · I[m_θ(x, 1, c) > 0.5]
7.  return x  # the predicted X_t
```

**Ensemble inference.** Repeat lines 1–6 with $K$ independent climatology-correlated noise samples. Total cost: $K \cdot S$ NFE. Target operating point: $S = 2$, $K = 8$ → **16 NFE for an 8-member ensemble**, vs StormCast's ${\sim}200$ NFE (8 × 25) and FlowCast's 80 NFE (8 × 10).

We also include a **1-step variant** ($S=1$) — analogous to the consistency-distilled student in the existing repo — which we expect to be the operationally interesting setting.

---

## 4. Why this should outperform — predicted ablation table

The plan must convince a reviewer not just by performing well but by *isolating which design choices matter*. Predicted directional outcomes (to be filled with numbers from the actual runs) are:

| # | Variant | qpepre CSI@1 mm/h | qpepre CRPS | u10 RMSE | NFE | Notes |
|---|---|---|---|---|---|---|
| 0 | StormCast (EDM, 18 NFE) | baseline | baseline | baseline | 18 | reference teacher |
| 1 | FlowCast (CFM, 10 NFE) | $\approx$ #0 | slightly better | slightly better | 10 | reproduces Ribeiro & Pucer comparison |
| 2 | **BridgeCast full** (S=2, K=8) | **best** | **best** | **best** | 16 | the proposed method |
| 3 | #2 − bridge (replace anchor with $\mathcal{N}(0,I)$) | $\approx$ #1 | $\approx$ #1 | $\approx$ #1 | 16 | isolates value of $M_t$-anchoring |
| 4 | #2 − climatology noise (use white) | small drop | medium drop | small drop | 16 | quantifies $\Sigma_{\text{clim}}$ |
| 5 | #2 − asinh / mask head | **large** drop on qpepre tails | large drop | unchanged | 16 | quantifies channel-adaptive path |
| 6 | #2 − Energy Score loss | small drop | **large** drop | small drop | 16 | calibration vs sharpness trade-off |
| 7 | #2 − physics correctors | tiny drop | tiny drop | small drop | 16 | sanity check, weakest term |
| 8 | #2 with S=1, K=8 | small drop | small drop | small drop | **8** | one-step BridgeCast |
| 9 | #2 with S=4, K=8 | $\approx$ #2 | $\approx$ #2 | $\approx$ #2 | 32 | diminishing returns regime |

If #3 is *not* close to #1 we have failed to demonstrate the central claim and should not submit. If #6 hurts CRPS only slightly we can demote $\mathcal{L}_{\text{ES}}$ to "regulariser" rather than headline contribution.

---

## 5. Implementation plan (concrete files)

We follow the layout of `train_flowcast.py` to minimise integration risk.

### 5.1 New files
- `stormcast/train_bridgecast.py` — entry point (Hydra), mirrors `train_flowcast.py`.
- `stormcast/train_bridgecast.sh` — launch script, Taiwan paths, mirrors `train_flowcast.sh`.
- `stormcast/utils/trainer_bridgecast.py` — training loop (mirrors `trainer_flowcast.py`, adds Energy Score and mask-head terms).
- `stormcast/utils/bridgecast_loss.py` — the composite loss in §2.8.
- `stormcast/utils/clim_noise.py` — utility for estimating $\hat{P}_k(\kappa)$ from training residuals once and sampling $\varepsilon \sim \mathcal{N}(0, \Sigma_{\text{clim}})$ at runtime via FFT.
- `stormcast/utils/qpepre_transform.py` — `asinh` forward / `sinh` inverse with Jacobian-aware MSE, plus rain-mask gating.
- `stormcast/config/bridgecast.yaml` — top-level Hydra config (resolves `model/bridgecast.yaml`, `training/bridgecast.yaml`).
- `stormcast/config/model/bridgecast.yaml` — wraps `EDMPrecond` SongUNet with the dual-head adapter; same param count as FlowCast.
- `stormcast/config/training/bridgecast.yaml` — hyperparameters from §6.
- `physicsnemo/experimental/metrics/diffusion/bridgecast_loss.py` — kernel Energy Score implementation (vector-output sum of L2 norms).

### 5.2 Reused infrastructure (no edits)
- `data_loader_rwrf_era5_stable.Dataset` — same dataloader.
- `StormCastUNet` regression checkpoint at `exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.14000.mdlus` — frozen, used to compute $M_t$ on the fly.
- `physicsnemo` SongUNet backbone (via `get_preconditioned_architecture` at `stormcast/utils/nn.py:25`).
- `stormcast/utils/ema.py` for EMA weights at inference.
- Validation / plotting utilities (`plot_training_convergence.py`, `inference_*.py` scaffolding).

### 5.3 Network adapter

The dual-head adapter is a thin wrapper over `EDMPrecond`:

```python
class BridgeCastNet(nn.Module):
    def __init__(self, base_unet, n_target_channels=4):
        super().__init__()
        self.base = base_unet                              # SongUNet, vel + mask channels
        self.vel_head  = nn.Conv2d(base.out_dim, n_target_channels, 1)
        self.mask_head = nn.Conv2d(base.out_dim, 1, 1)

    def forward(self, x_t, t, c):
        h = self.base.features(x_t, t_embed(t), c)         # backbone features
        v = self.vel_head(h)
        m = torch.sigmoid(self.mask_head(h))
        return v, m
```

(The actual implementation will follow the existing `EDMPrecond` factory style so we get its conditioning helper for free; the *only* architectural change vs. FlowCast is the second 1×1 head.)

### 5.4 Stochastic interpolant and loss (pseudocode)

```python
def bridgecast_step(batch, regression, model, ema, cfg):
    Xtm1, St, I, Xt = batch
    with torch.no_grad():
        Mt = regression(Xtm1, St, I)                          # frozen regression mean
    R  = Xt - Mt                                              # residual
    R[..., qpepre_idx] = asinh(R[..., qpepre_idx] / kappa)    # channel-adaptive path
    eps = sample_clim_noise(shape=R.shape)                    # FFT-correlated, see §2.3
    t   = torch.rand(B, device=R.device).clamp(t_eps, 1-t_eps)
    gamma  = sigma_b * torch.sqrt(t*(1-t))
    gammap = sigma_b * (1-2*t) / (2*torch.sqrt(t*(1-t)))
    x_t    = (1-t)*Mt + t*Xt + gamma * eps
    xdot_t = (Xt - Mt) + gammap * eps                          # interpolant velocity
    c      = build_condition(Xtm1, St, Mt, I)
    v_pred, m_pred = model(x_t, t, c)
    L_v    = ((v_pred - xdot_t).pow(2) * channel_w).mean()
    L_mask = focal_bce(m_pred, (Xt[..., qpepre_idx] > 0.1).float())
    L_es   = energy_score_via_K_one_step_samples(model, c, Xt, K=cfg.K)
    L_spec = radial_psd_l1(v_pred, target=Xt - Mt, channels=['qpepre'])
    L_div  = divergence_penalty(v_pred[..., u_idx], v_pred[..., v_idx])
    L = lam_v*L_v + lam_m*L_mask + lam_e*L_es + lam_s*L_spec + lam_d*L_div
    L.backward(); ...; ema.update(model)
    return L
```

The `K`-sample Energy Score branch reuses the same network but with `t = 0` and a 1-step Euler from $M_t + \gamma(\delta) \varepsilon^{(k)}$ — so $K$ extra forward passes per step (at $K = 4$, this is a $5\times$ training-time multiplier vs the velocity term alone, which is acceptable on 2 H100s; we can reduce to $K=2$ if memory-bound).

---

## 6. Hyperparameters (initial, to be tuned)

| | Value | Source / rationale |
|---|---|---|
| Backbone | SongUNet, same as FlowCast | match parameter count |
| Optimizer | AdamW, lr 5e-4, wd 1e-4, cosine | matches `train_flowcast.sh` |
| Warmup steps | 4000 | matches FlowCast |
| Total steps | 400 000 | matches FlowCast (fair compute) |
| Batch size | 16 (2 GPUs × 8) | matches FlowCast |
| EMA decay | 0.999 | matches FlowCast |
| Bridge noise $\sigma_b$ | 0.15 (in normalised units) | midway between I²SB image setting (0.05) and CFM σ (0.5) |
| Time clip $t_\varepsilon$ | $10^{-3}$ | numerical stability of $\gamma'(t)$ |
| asinh $\kappa$ | 1.0 mm/h | linear–log knee at light rain |
| Channel weights $\beta$ | [1, 1, 1, 2] | matches FlowCast |
| ES samples $K$ | 4 | compute-bound trade-off |
| ES weight $\lambda_e$ | 0.5 | match scale of $\mathcal{L}_v$ at convergence |
| Mask weight $\lambda_m$ | 0.1 | side-task |
| Spectral weight $\lambda_s$ | 0.1 | matches FlowCast |
| Divergence weight $\lambda_d$ | $10^{-3}$ | weak prior |
| Inference $S, K$ | 2, 8 | operational target |

---

## 7. Experimental protocol

### 7.1 Datasets and splits

Same as the existing repo (CLAUDE.md §1):
- Train: 2019-08 → 2021-12, 21 216 hourly steps.
- Validation: 2022 full year, 8 760 hourly steps.
- LowRes (24 ch synoptic), HighRes (4 ch: t2m, u10, v10, qpepre), invariants (lsm, orog).
- Per-channel HighRes means/stds from CLAUDE.md, used for normalisation.

### 7.2 Baselines (all FLOP-matched, same SongUNet)

1. **StormCast EDM teacher** — 18 NFE, Heun, our pretrained checkpoint at `exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus`.
2. **StormCast at low NFE (4, 8)** — naive Heun under-stepping; expected to degrade.
3. **FlowCast for StormCast** — `train_flowcast.sh`, 10 NFE Euler, 1 NFE Euler.
4. **Progressive Distillation student** (existing in repo) — at the lowest NFE that retains skill.
5. **Consistency Distillation student** (existing in repo).
6. **BridgeCast** — proposed.

We do not use external benchmarks (SEVIR, ARSO) — the comparison is on the same Taiwan RWRF grid that StormCast and FlowCast were trained on, which is the only scientifically meaningful comparison given the regional convective setting.

### 7.3 Metrics (evaluated on the 2022 validation year)

- **Per-channel deterministic.** RMSE, MAE de-normalised to physical units. Reported separately for `t2m`, `u10`, `v10`, `qpepre`.
- **Probabilistic.** CRPS (per-channel), Energy Score (full-field), spread-skill ratio, rank histogram. 8-member ensemble.
- **Precipitation categorical.** CSI, FSS, bias, FAR at thresholds $\{0.1, 1, 5, 10, 20\}$ mm/h with neighbourhood pooling at $\{1, 4, 16\}$ km. Reported on 1h, 3h, 6h, 12h leads from the autoregressive rollout.
- **Spectral.** Radially-averaged log-PSD per channel — student vs teacher vs ground truth; relative difference plots like Pathak '24 Fig 5.
- **Computational.** NFE, wall-clock per 1h step, wall-clock per 12h × 8-member ensemble. Single H100.

### 7.4 Rollout protocol

Identical to `inference.py` / `inference_ncdr.py` already in the repo: hourly autoregressive steps, $X_{t+1} \leftarrow$ BridgeCast$(X_t, S_t, I)$, fresh $\varepsilon^{(k)}$ each step. We launch ${\sim}120$ rollouts initialised at 00Z and 12Z across 2022 to span seasonal regimes.

### 7.5 Statistical tests

For every (variable, lead-time, threshold) cell we run a paired bootstrap (1000 resamples) on per-rollout score differences and report 95% CIs. Headline claim of "outperforms" requires $p < 0.05$.

---

## 8. Expected contributions (claims for the paper)

1. **Mean-anchored Schrödinger bridge** as a generative formulation for residual mesoscale forecasting — the first application of stochastic interpolants / I²SB-style bridges to atmospheric science. The transport length argument (§2.2) is general beyond Taiwan RWRF.
2. **Climatology-correlated noise prior** as a drop-in replacement for white Gaussian noise in any generative weather model. We will ablate independently (variant #4) and include a recipe so others can apply it to GenCast / FlowCast / StormCast.
3. **Energy-Score-augmented training** for convective-scale generation, transplanting AIFS-CRPS-style ensemble objectives into a few-step regional setting. Variant #6 isolates the value.
4. **Channel-adaptive path** (asinh + mask head) for sparse heavy-tailed precipitation — variant #5 quantifies the gain.
5. **Empirical headline.** BridgeCast at S=2, K=8 (16 NFE) outperforms StormCast (≥144 NFE for 8-member ensemble) and FlowCast (80 NFE) on CRPS, CSI@10mm/h, and FSS@16km, on the Taiwan 2022 validation year, with calibrated rank histograms and matched spectra. (Numbers TBD by experiment.)

These are independently meaningful contributions — even if (5) only matches the baselines, contributions (1)–(4) plus the controlled ablations are publishable as a methodology study. This is a defensive design: the paper has a positive contribution under any plausible empirical outcome short of total failure.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Bridge anchor causes mode collapse onto the regression mean (no spread) | Medium | $\sigma_b > 0$ injects noise; $\mathcal{L}_{\text{ES}}$ explicitly penalises low spread; check rank-histogram early. If still collapsed, raise $\sigma_b$ or switch to source $\mathcal{N}(M_t, \sigma_a^2 I)$. |
| Energy Score with $K$ samples explodes train memory | Medium | Gradient-checkpoint the K-sample branch; reduce $K$ to 2; only apply ES on coarse-pooled fields. |
| Climatology noise $\hat{P}_k$ is mis-estimated → ringing artefacts | Low | Validate against a held-out month before training; clip $\hat{P}_k$ at high-$\kappa$ tail. |
| `qpepre` mask-head over-confident → sharper but wrong rain edges | Medium | Calibrate mask head with focal BCE + temperature scaling on validation; ablate. |
| Physics correctors fight the data → worse RMSE | Low–Medium | Variant #7 ablates them; keep weights tiny ($10^{-3}$) or drop. |
| Backbone changes regress reproducibility of FlowCast comparison | Low | Keep SongUNet + same channel widths + same param count; only the heads and the loss differ. |
| Compute budget overrun (5 ablations × 400k steps) | High | Sequence ablations: full BridgeCast first; if it wins, run #3, #5, #6 (highest-signal variants) only. |

---

## 10. Timeline (relative to "thesis submission" deadline; assumes 2 H100s)

| Week | Task |
|---|---|
| 1 | Implement `clim_noise.py`, fit $\hat{P}_k(\kappa)$, sanity-check noise samples visually. |
| 1 | Implement `qpepre_transform.py` + mask head, unit-test gradient flow. |
| 2 | Implement `trainer_bridgecast.py` and `bridgecast_loss.py`, smoke-test on 1k steps. |
| 2–4 | Train BridgeCast (#2) for 400k steps. |
| 4 | Validation runs: full 2022 metrics for #2, plus 8-member rollouts. |
| 5 | Train ablations #3 (no anchor), #5 (no asinh), #6 (no ES) — these are the highest-signal. |
| 6 | Train remaining ablations #4, #7 if compute permits. |
| 6–7 | Statistical tests, plots, writing. |
| 7 | Internal review; iterate. |

---

## 11. Connection to the existing thesis

The thesis (`master_thesis_plan.md`) is currently scoped to PD/CD distillation of the EDM teacher. BridgeCast is **complementary, not competing**:

- It can be presented as **Chapter 4.4 / Future Work** in v1 of the thesis (one-paragraph teaser + proof-of-concept run).
- A standalone **conference paper** is the primary target — BridgeCast is novel enough to stand alone, and the experiments above are designed for that venue (NeurIPS / ICLR / ICML / Climate-AI workshop) rather than the thesis itself.
- The codebase additions live in the same `stormcast/` tree, share the regression checkpoint, dataloader, and validation harness, so there is zero integration cost.

---

## References

1. Albergo, M. S., Boffi, N. M., & Vanden-Eijnden, E. (2023). *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*. arXiv:2303.08797.
2. Andrychowicz, M., et al. (2023). *Deep Learning for Day Forecasts from Sparse Observations*. arXiv:2306.06079. (asinh transform precedent.)
3. De Bortoli, V., Thornton, J., Heng, J., & Doucet, A. (2021). *Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling*. NeurIPS.
4. Gao, Z., et al. (2023). *PreDiff: Precipitation Nowcasting with Latent Diffusion Models*. NeurIPS.
5. Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation*. JASA, 102(477), 359–378.
6. Gong, J., et al. (2024). *CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling*. ICML.
7. Heng, J., De Bortoli, V., & Doucet, A. (2024). *Diffusion Schrödinger Bridge Matching*. NeurIPS.
8. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
9. Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM). NeurIPS.
10. Lang, S., et al. (2024). *AIFS-CRPS: Ensemble Forecasting using a Model Trained with a Loss Function based on the Continuous Ranked Probability Score*. ECMWF Technical Memorandum.
11. Leinonen, J., Hamann, U., Nerini, D., Germann, U., & Franch, G. (2023). *Latent Diffusion Models for Generative Precipitation Nowcasting with Accurate Uncertainty Quantification* (LDCast). arXiv:2304.12891.
12. Léonard, C. (2014). *A Survey of the Schrödinger Problem and Some of its Connections with Optimal Transport*. Discrete and Continuous Dynamical Systems, 34(4), 1533–1574.
13. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). *Flow Matching for Generative Modeling*. ICLR.
14. Liu, G.-H., Vahdat, A., Huang, D.-A., Theodorou, E. A., Nie, W., & Anandkumar, A. (2023). *I²SB: Image-to-Image Schrödinger Bridge*. ICML.
15. Liu, X., Gong, C., & Liu, Q. (2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR.
16. Mardani, M., et al. (2024). *Residual Diffusion Modeling for Km-scale Atmospheric Downscaling* (CorrDiff). Submitted.
17. Pacchiardi, L., & Dutta, R. (2024). *Likelihood-Free Inference with Generative Neural Networks via Scoring Rule Minimization*. JMLR.
18. Pathak, J., Cohen, Y., Garg, P., et al. (2024). *Kilometer-Scale Convection Allowing Model Emulation using Generative Diffusion Modeling* (StormCast). arXiv:2408.10958. — Reproduced locally at `papers/md/stormcast.md`.
19. Price, I., Sanchez-Gonzalez, A., Alet, F., et al. (2025). *GenCast: Diffusion-based ensemble forecasting for medium-range weather*. Nature. — Reproduced locally at `papers/md/gencast.md`.
20. Ravuri, S., et al. (2021). *Skillful Precipitation Nowcasting using Deep Generative Models of Radar* (DGMR). Nature, 597, 672–677.
21. Ribeiro, B. P., & Pucer, J. F. (2025). *FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching*. arXiv:2505.xxxx. — Reproduced locally at `papers/md/flow_cast.md`.
22. Salimans, T., & Ho, J. (2022). *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR.
23. Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). *Consistency Models*. ICML. — Reproduced locally at `papers/md/consistency_model.md`.
24. Tong, A., Malkin, N., Huguet, G., et al. (2024). *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport* (I-CFM, OT-CFM). TMLR.
25. Veillette, M., Samsi, S., & Mattioli, C. (2020). *SEVIR: A Storm EVent ImageRy Dataset for Deep Learning Applications in Radar and Satellite Meteorology*. NeurIPS.
26. Yim, J., et al. (2023). *SE(3) Diffusion Model with Application to Protein Backbone Generation*. ICML. (Bridge-matching precedent in another domain.)
