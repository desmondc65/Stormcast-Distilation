# Why the current architecture is optimal for next-hour prediction — and where the novelty actually lives

**Purpose.** This note answers the advisor's "this lacks novelty / make a bigger
architectural change" critique. It argues two things, separately, because they
are different claims:

1. **The two-stage architecture (frozen regression mean μ + a generative
   residual head on a SongUNet backbone) is the *right* design for next-hour
   prediction** — not an arbitrary inherited choice. Replacing the backbone
   would, on this problem, more likely hurt than help, and would add confounds
   to the one comparison the thesis actually rests on.
2. **The novelty should not — and need not — come from the backbone.** A bare
   "diffusion → flow matching" swap is, by itself, thin methodological novelty,
   and the honest move is to concede that. The thesis-level contribution belongs
   in **how the flow is constructed on top of the fixed architecture** (the
   anchored μ→M bridge of [flowcast_improvement.md](flowcast_improvement.md),
   §Direction 1). That reframes the advisor's "bigger idea" as a *generative-
   process* change rather than a network change — more novel **and** lower-risk
   than a DiT/Mamba/Earthformer swap.

The structure: §1 states the claim precisely; §2 is the physics/information
argument for the two-stage decomposition; §3 is the ML argument for the SongUNet
backbone; §4 is why flow matching is the right *objective* on it; §5 is an honest
novelty audit; §6 is the reconciliation (the bridge) and a script for the
conversation with the advisor.

---

## 1. State the claim precisely

"Optimal" in the strict mathematical sense is unprovable and not what we mean.
The defensible claim is **optimality-in-context**: the architecture is matched to
four constraints that next-hour regional prediction actually imposes.

| Constraint of the problem | Architectural consequence | Where this is justified |
|---|---|---|
| At Δt = 1 h, predictability splits into a large deterministic part + a small stochastic residual | Two-stage: cheap deterministic μ, generative model only on the small residual | §2 |
| The residual is spatially local, multi-scale, near-translation-equivariant | Convolutional U-Net with skip connections + multi-resolution encoder/decoder | §3.1 |
| Small regional dataset (~18.8 k training samples, 192×96) | CNN inductive bias (locality, equivariance) generalises; transformers are data-hungry | §3.2 |
| Small domain (4°×2.5°, 192×96) | 5 downsamplings already cover the domain → no need for global attention | §3.3 |

The two-stage residual design is exactly NVIDIA StormCast's (Pathak et al. 2024)
and is encoded here in [stormcast/utils/nn.py](stormcast/utils/nn.py): a frozen
`StormCastUNet` regression net produces μ; a `EDMPrecond` / `FlowCastPrecond`
SongUNet learns the residual `r = M_{t+1} − μ_{t+1}` (see
[build_network_condition_and_target](stormcast/utils/nn.py#L100)). The thesis
keeps that scaffold and varies only the *generative objective* on the residual.

---

## 2. Why the two-stage decomposition is right *at the 1-hour lead time*

This is the load-bearing argument, and it is genuinely about the physics of
short-lead prediction — not a post-hoc rationalisation of an inherited design.

### 2.1 Predictability separates by magnitude and scale at Δt = 1 h

Atmospheric predictability is lead-time- and scale-dependent. One hour ahead, the
state is dominated by **persistence and advection** of structures that already
exist in `M_t`. The synoptic and mesoscale fields (temperature, the wind bulk)
are nearly deterministic at this horizon; what is genuinely *uncertain* is
confined to (a) the heavy-tailed, sparse precipitation channel and (b) the
smallest convective spatial scales. Concretely, in this repo:

- The regression mean μ already explains most of the variance of the slow
  channels. The discussion note states it directly: *"t2m is dominated by the
  regression mean … μ already explains most of the variance and the residual
  head has little to do"* ([experiment_scripts/discussion.md](experiment_scripts/discussion.md), §2.5).
- The residual `r = M_{t+1} − μ_{t+1}` is therefore **small in magnitude,
  near-zero-mean, and concentrated in qpepre + high spatial frequencies** — which
  is exactly why FlowCast standardises it to unit scale (`x_1 = r/σ_data`) and
  why the per-channel weight + spectral term target qpepre specifically
  ([stormcast/utils/flowcast_loss.py](stormcast/utils/flowcast_loss.py)).

### 2.2 That separation is *why* splitting the model is optimal, not just convenient

Two formal ways to see it, both pointing the same direction:

- **Control-variate / variance reduction.** μ is a near-optimal, cheap control
  variate for `M_{t+1}`. Modelling `M_{t+1} = μ + r` and learning only the
  distribution of `r | c` is a strictly lower-variance estimation problem than
  modelling the full field `M_{t+1} | c`, because μ removes the large
  deterministic component before the generative model ever sees it. The
  generative head spends all its capacity on the *conditional residual
  distribution*, which has far lower entropy than the full-state distribution.
- **Capacity allocation.** A deterministic regression solves the easy ~95% of
  the variance with an MSE objective (cheap, stable, no sampling). A single
  end-to-end generative model over the full state would have to *re-derive* that
  deterministic bulk through a much harder stochastic objective — wasting
  capacity and sample budget to relearn what a regression nails for free.

### 2.3 Contrast with the regime where a "bigger generative architecture" *is* justified

End-to-end generative weather models over the full state (GenCast-style, multi-day
ensembles) are justified **because at multi-day lead times the chaotic component
dominates and the deterministic mean is a poor predictor** — there the residual is
*not* small and the two-stage trick buys little. Next-hour regional prediction is
the opposite regime: the deterministic mean is excellent, the residual is small,
and a two-stage decomposition is matched to that structure. So "use a bigger
single generative architecture" is borrowing a design from a *different
predictability regime*. At Δt = 1 h it solves a harder problem than the physics
requires.

**This is the cleanest rebuttal to "make a bigger architectural change":** a
bigger end-to-end model is not more correct here — it is mismatched to the
short-lead predictability structure. The two-stage design is the inductive bias,
and it is the *right* one for this lead time.

---

## 3. Why the SongUNet backbone is near-optimal for the residual

Given that we are modelling a small, multi-scale, local residual field, the
convolutional U-Net is close to the best available choice — and the alternatives
the advisor might have in mind (DiT, Mamba, Earthformer) are worse *for this
problem*, not better.

### 3.1 The residual matches what a U-Net is good at

The residual field is spatially local, multi-scale, and locally
translation-equivariant. The SongUNet's encoder/decoder over 5 resolution levels
(`channel_mult=[1,2,2,2,2]`) captures the multi-scale spectrum, and the skip
connections preserve the fine detail that sharp precipitation cores need — which
is precisely the failure mode (over-smoothed, "always slightly damp" qpepre) that
the thesis is fighting (CLAUDE.md §8). The architecture's inductive bias is
aligned with the target's statistics.

### 3.2 The data scale argues *for* the CNN bias, *against* transformers

The cleaned dataset is ~18.8 k training samples on a 192×96 grid — small by
deep-learning standards. A CNN's built-in translation equivariance and locality
are exactly the priors that let it generalise from limited data. A DiT or a
sequence model must **learn** those priors from data and is well known to need
much more of it to match a CNN at this scale. Swapping the backbone for a
transformer here trades a free, correct inductive bias for a data-hungry one —
the likely outcome on ~18.8 k samples is *worse*, not better.

### 3.3 The domain size removes the usual reason to add attention

All shipped configs use `attn_resolutions=[]` — no self-attention except the one
fixed bottleneck block ([model_parameters.md](model_parameters.md) §6). That is
justified: at 192×96 over a 4°×2.5° box, five downsamplings (192→96→48→24→12)
give the bottleneck a receptive field that already spans the whole domain. There
is no long-range dependency that local convolution + downsampling cannot capture,
so global attention would add parameters and data-hunger for no expected gain.
(If a future, much larger domain changed this, *that* would be the moment to add
attention — and the factory already supports it via `attn_resolutions`.)

### 3.4 Parameter parity makes the experiment clean — a methodological *strength*

All three generative heads instantiate the **identical** 78.7 M-parameter
SongUNet with the same 14→4 channel contract; the regression net is 78.4 M
([model_parameters.md](model_parameters.md) §TL;DR). EDM and FlowCast differ
*only* in the loss/sampler math — the preconditioner wrappers add zero learnable
parameters. This is not a footnote; it is what makes the headline comparison
valid:

> Because the backbone, the conditioning bundle, the frozen μ, the dataset, and
> the parameter count are all held fixed, the FlowCast-vs-EDM result isolates the
> **objective**. Change the backbone and you reintroduce exactly the confound
> that this controlled design removes.

So "keep the architecture fixed" is not conservatism — it is the experimental
control that lets the thesis make a clean causal claim about flow matching vs
score matching.

---

## 4. Why flow matching is the right *objective* on this fixed architecture

The architecture is fixed; the question the thesis answers is "what generative
objective extracts the most skill from it per unit of data and per unit of
inference cost?" The evidence already in the repo says: flow matching.

At a matched ~2 M-training-sample budget, same μ, same backbone
([experiment_scripts/result_table.md](experiment_scripts/result_table.md) Table A;
[experiment_scripts/discussion.md](experiment_scripts/discussion.md) §1):

| Axis | EDM teacher | FlowCast student | Source |
|---|---|---|---|
| Inference cost / 12 h rollout | 24.5 s (Heun-18, ~36 NFE) | **8.4 s (Euler-10)** ≈ 3× cheaper | discussion §1 |
| Training cost to 2 M samples | ~28–37 GPU-h | **~28 GPU-h** | result_table "GPU-hours" |
| FAR-M (false-alarm rate) | 0.696 | **0.428** (≈ halved) | discussion §2.1 |
| FSS-P16-M (heavy-rain coherence) | 0.0009 | **0.0068** (≈ 7×) | discussion §2.2 |
| qpepre / wind RMSE | baseline | **5–20 % lower** | discussion §1 |
| t2m RMSE | **1.04** | 1.17 (≈ 10 % worse) | discussion §2.5 |
| Parameters | 78.7 M | 78.7 M (identical) | model_parameters.md |

Two of these have a clean, intrinsic explanation (not just recipe tuning),
spelled out in [discussion.md](experiment_scripts/discussion.md) §3:

- **Lower-variance gradient.** I-CFM regresses a fixed unit-scale velocity
  `u = x_1 − x_0`; EDM must denoise the same residual across ~3 decades of σ with
  a heavy-tailed reweighting `w(σ) ∝ σ^-2`. Lower per-sample gradient variance →
  more learned per step → better skill at a fixed sample budget (§3.2).
- **Straighter sampler trajectory.** The CFM conditional path is a straight line,
  so Euler-10 already approximates it well; EDM's PF-ODE is curved and needs
  Heun-18. This is *why* "more NFE doesn't help" FlowCast (NFE 10 ≥ 15 ≥ 20 on
  FAR/FSS, discussion §2.4) — the cheapest configuration is also the most
  skilful. That is a strict deployment win, not a trade-off.

These two are properties of the *loss-and-trajectory geometry*, independent of
the architecture — which is exactly why the architecture should stay fixed while
the objective is the variable under study.

---

## 5. Honest novelty audit — what is thin, what is real

A defense survives on honesty about this table, not on overclaiming. Separate the
contribution into tiers:

| Tier | Contribution | Novelty | Honest verdict |
|---|---|---|---|
| Thin | "Use I-CFM instead of EDM" | Known (Lipman 2023, Tong 2024) | **The advisor is right that this alone is engineering, not research.** Do not lead with it. |
| Recipe | qpepre-aware residual CFM: per-channel velocity weighting + radial log-PSD spectral term on the heavy-tailed precip channel ([flowcast_loss.py](stormcast/utils/flowcast_loss.py)) | Not in the FlowCast paper; targeted at this dataset's hard channel | Real but modest. Responsible for a measurable slice of the FSS/FAR win (discussion §2.2, §3.7). |
| Application | First CFM-vs-diffusion head-to-head on **regional km-scale hourly NWP residuals** (Taiwan RWRF), parameter-matched and confound-audited | Regime/domain not previously tested | Real. Good science; modest as a *methods* claim. |
| Deployment | Equal-or-better skill at ~3× lower inference NFE and equal training cost, with the non-obvious "more NFE ≠ better" finding | — | Real, useful, but it is an efficiency result, not a new method. |

The repo's own discussion is admirably honest that part of the win is *removable*
recipe (qpw=2.0, spectral term, EMA asymmetry — discussion §3.7, §5). A defense
that hides that gets caught; a defense that states it and isolates the
*irreducible* algorithmic part (§3.2–§3.6) is far stronger.

**Conclusion of the audit:** the architecture work is *correct and well-motivated*
(§2–§4), and that is defensible on its own as the experimental foundation — but
the advisor is not wrong that the *generative idea*, as currently stated, is a
known swap dressed for a new domain. The fix is §6.

---

## 6. The reconciliation: keep the architecture, move the novelty into the flow

The advisor wants a "bigger idea." The instinct to keep the architecture is
correct (§2–§4). These are **not in conflict**, because the architecture is not
where the idea should live. The genuinely novel move is to change **how the flow
is constructed on top of the fixed SongUNet** — which is precisely what
[flowcast_improvement.md](flowcast_improvement.md) already lays out:

- **Direction 1 — Anchored stochastic-interpolant bridge (μ → M, not noise →
  residual).** Today μ is just a conditioning channel and a target shift. The
  bridge makes μ the **structural prior endpoint of the flow**: the ODE starts at
  `x_0 = μ + σ_prior·ε` and integrates to `x_1 = M_{t+1}` directly (Albergo &
  Vanden-Eijnden 2023 stochastic interpolants; I²SB, Liu 2023). The flow is
  *short* because the endpoints are physically close, and it directly exploits
  the residual structure of next-hour prediction that §2 established. This is the
  chapter that gives the thesis its title — and it runs on the **identical**
  backbone (`name="bridge"` is literally `FlowCastPrecond` again,
  [nn.py:70](stormcast/utils/nn.py#L70); same 78.7 M parameters,
  [model_parameters.md](model_parameters.md)). It is already partly implemented
  ([bridge.md](bridge.md)).
- **Direction 2/3 — heavy-tailed / channel-aware prior on qpepre, and OT-CFM
  minibatch coupling** straighten and shorten the flow further; both keep the
  backbone untouched.
- **Direction 5 — physics-aware projection** (divergence-free wind inside the ODE
  integrator) is a *constraint at integration time*, again with no backbone
  change.

Every one of these is a change to the **generative process**, none is a change to
the **network**. That is the whole point:

> The architecture is fixed *because* it is matched to next-hour predictability
> (§2), to the residual's statistics (§3.1), to the data scale (§3.2), and to the
> domain size (§3.3), and *because* fixing it is the experimental control that
> makes the objective comparison clean (§3.4). The novelty the advisor is asking
> for belongs in the flow construction — μ→M bridge, anchored prior,
> OT coupling, physics-projected ODE — which is both **more novel** than a
> backbone swap and **lower-risk**, since it leaves the validated scaffold intact.

### What to actually say to the advisor

1. **Concede the fair point.** "You're right that I-CFM-for-EDM, stated plainly,
   is a known swap. I'm not resting the contribution on that."
2. **Defend the architecture on physics, not preference.** "The two-stage
   residual design is matched to *next-hour* predictability — the deterministic
   mean carries ~95% of the variance and the residual is small, sparse, and
   multi-scale (here is the t2m-is-solved-by-μ evidence). A bigger end-to-end
   generative model is borrowed from the multi-day chaotic regime and solves a
   harder problem than this lead time needs."
3. **Defend keeping it fixed as a control.** "Holding the 78.7 M-parameter
   backbone, μ, and data fixed is what lets me make a clean causal claim about the
   *objective*. Changing the backbone reintroduces the confound."
4. **Offer the bigger idea where it belongs.** "The methodological contribution
   is the anchored μ→M stochastic-interpolant bridge: μ stops being a conditioning
   channel and becomes the prior endpoint of the flow. That reformulates the
   generative process to exploit the residual structure of short-lead prediction
   — on the same optimal backbone. Plus OT coupling and a heavy-tailed qpepre
   prior as the supporting chapters."

That answer keeps the architecture the advisor is implicitly questioning,
concedes the part of the critique that is fair, and hands back a genuinely bigger
idea — sited exactly where novelty in a residual-flow NWP model *should* be.

---

### Pointers

- Architecture & factory: [stormcast/utils/nn.py](stormcast/utils/nn.py)
- Parameter parity (78.7 M, all heads identical): [model_parameters.md](model_parameters.md)
- Head-to-head results + honest caveats: [experiment_scripts/discussion.md](experiment_scripts/discussion.md), [experiment_scripts/result_table.md](experiment_scripts/result_table.md)
- The "bigger idea" menu (all backbone-preserving): [flowcast_improvement.md](flowcast_improvement.md)
- Bridge already walked end-to-end: [bridge.md](bridge.md)
- Project wiki: [CLAUDE.md](CLAUDE.md)
</content>
</invoke>
