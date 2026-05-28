# Conditional Flow Matching and Rectified Flow

A reading-room note. The goal is to make precise what FlowCast is doing under
the hood and how it relates to the wider flow-matching / rectified-flow
literature.

References:
- Lipman, Y. et al. *Flow Matching for Generative Modeling*. ICLR 2023.
- Tong, A. et al. *Improving and Generalizing Flow-Based Generative Models with
  Minibatch Optimal Transport*. TMLR 2024. (Introduces I-CFM.)
- Liu, X., Gong, C., Liu, Q. *Flow Straight and Fast: Learning to Generate and
  Transfer Data with Rectified Flow*. ICLR 2023.
- Ribeiro & Pucer 2025, *FlowCast* — application paper this repo follows.

---

## 1. The setup: learn a transport ODE

We want to sample from a data distribution `q(x_1)` (clean image, latent code,
residual, …). We pick a tractable prior `p(x_0) = N(0, I)`. A **continuous
normalizing flow** is an ODE

```
dx/dt = v_θ(x, t),   x(0) ~ p_0,   t ∈ [0, 1]
```

whose flow `φ_t` pushes `p_0` to a target `p_1 ≈ q`. Training the velocity
field `v_θ` directly by maximum likelihood (the old CNF route) requires
solving the ODE *during training* to compute log-determinants — expensive for
high-dim spatiotemporal data.

Flow matching sidesteps the likelihood and learns `v_θ` by **regression**
against a target vector field. The only question is: what target?

---

## 2. The intractable target: the marginal vector field

Imagine some smooth probability path `p_t(x)` that interpolates `p_0` and
`p_1` (say, by adding less noise as `t` grows). There is a unique
deterministic velocity field `u_t(x)` whose flow generates that path — the
**marginal vector field**. If we could regress `v_θ` against it,

```
L_FM = E_{t, x ~ p_t} || v_θ(x, t) − u_t(x) ||²
```

we'd be done. The catch: `u_t(x)` is defined by a marginalization over the
data distribution and is **not available in closed form**.

---

## 3. The CFM trick: condition on a data sample

Pick a **conditional** probability path `p_t(x | x_1)` for each data sample
`x_1`. Two requirements:

1. `p_0(x | x_1) = p(x_0)` (prior at `t=0`).
2. `p_1(x | x_1) = δ(x − x_1)` (a Dirac at `t=1`, or a tight Gaussian).

Then each conditional path has its own **conditional vector field**
`u_t(x | x_1)` that is *tractable* — you choose `p_t(·|x_1)` to make it so.
The marginal field is the data-conditioned mixture

```
u_t(x) = E_{x_1 | x} [ u_t(x | x_1) ].
```

Lipman et al.'s key theorem: regressing against the *conditional* field gives
the same gradient in expectation as regressing against the *marginal* one.

```
L_CFM = E_{t, x_1 ~ q, x ~ p_t(·|x_1)} || v_θ(x, t) − u_t(x | x_1) ||²
```

Same optimum, no marginalization. Crucially: **no ODE integration during
training** — you just sample `t`, draw `(x_0, x_1)`, build `x_t` from the
chosen conditional path, and regress.

This is what "simulation-free" means in the flow-matching literature.

### Gaussian conditional paths

The standard family: `p_t(x | x_1) = N(x; μ_t(x_1), σ_t² I)` with smooth
schedules `μ_t, σ_t`. The conditional vector field then has a closed form

```
u_t(x | x_1) = (μ̇_t − σ̇_t/σ_t · (μ_t − x_1)) + (σ̇_t/σ_t) · x.
```

Different schedules recover different prior models:
- Variance-exploding diffusion ⇒ score matching.
- VP diffusion ⇒ DDPM-style.
- **Linear interpolation between `x_0` and `x_1`** ⇒ rectified flow / I-CFM
  (next section).

This is why papers describe CFM as a *generalization* of diffusion: pick the
right Gaussian schedule and you recover EDM/DDPM training; pick the linear
one and you get straight-path flows.

---

## 4. Independent CFM (the FlowCast variant)

I-CFM (Tong et al. 2024) picks the most aggressive simplification:

- Draw `x_1 ~ q` and `x_0 ~ p_0` **independently** (no coupling).
- Conditional path is a straight line plus a small isotropic jitter:

  ```
  x_t = (1 − t)·x_0 + t·x_1 + σ·ε,    ε ~ N(0, I),   σ small.
  ```

- Differentiating the mean gives the conditional target

  ```
  u_t(x | x_0, x_1) = x_1 − x_0
  ```

  — a constant in `t`, independent of `x_t`. The student just regresses the
  straight-line velocity.

Algorithm 1 of FlowCast is exactly this:

```
t ~ U(0, 1),   x_0 ~ N(0, I),   ε ~ N(0, I)
x_t = (1-t) x_0 + t x_1 + σ ε
L = || v_θ(x_t, t, c) − (x_1 − x_0) ||²
```

The `σ > 0` term is the **thickening** trick: it widens the otherwise
singular conditional path (a Dirac line in `(x_0, x_1)` space) into a thin
Gaussian tube, which empirically stabilizes high-dim training. FlowCast uses
`σ = 0.01`.

In this repo: [stormcast/utils/flowcast_loss.py](stormcast/utils/flowcast_loss.py)
implements exactly this objective.

### Why `S` does not appear at training time

The student never integrates the ODE during training. It sees one random `t`
per minibatch and regresses a *pointwise* velocity. The number of ODE steps
`S` only enters the sampler (Algorithm 2): `Δt = 1/S`, Euler updates of
`x ← x + v_θ(x, t) · Δt`. Train once, sample at any `S`.

---

## 5. General CFM coupling vs the I-CFM specialization

Section 3 quietly introduced two design knobs in CFM that are easy to
conflate:

1. The **conditional path shape** `p_t(x | x_1)` — typically a Gaussian
   schedule with means `μ_t(x_1)` and stds `σ_t`. Section 3 already
   surveys this axis (linear, VE, VP, ...).
2. The **coupling** `π(x_0, x_1)` — the joint distribution from which
   paired noise/data samples are drawn during training. This axis is
   what "Independent" refers to in I-CFM, and it gets less air-time in
   most write-ups.

I-CFM picks the simplest possible answer for *both* knobs: linear path
with a tiny isotropic jitter, and independent product coupling
`π(x_0, x_1) = p_0(x_0) · q(x_1)`. To see what's being given up — and
what other points in design space look like — it helps to look at the
coupling spectrum that "general" CFM admits.

### 5.1 What a coupling is, formally

A **coupling** is any joint distribution `π` over `(x_0, x_1)` whose
marginals recover the prior and the data:

```
∫ π(x_0, x_1) dx_1 = p_0(x_0),    ∫ π(x_0, x_1) dx_0 = q(x_1).
```

Once you fix `π`, the CFM training loop becomes:

1. Sample `(x_0, x_1) ~ π`.
2. Sample `t ~ U(0,1)` and build the conditional interpolant `x_t`.
3. Regress `v_θ(x_t, t)` onto the conditional target
   `u_t(x_t | x_0, x_1)`.

Different choices of `π` keep the marginal `p_t(x)` identical at every
`t` (those marginals are pinned by definition), but they change the
shape of the **velocity field** that `v_θ` must learn. The reason is
geometric: the regression target `u_t(x_t | x_0, x_1)` depends on
*which* `x_1` is paired with the current `x_0`. A coupling that pairs
each noise sample with a *nearby* data sample yields conditional paths
that do not cross; a coupling that pairs them at random forces the paths
to cross, which forces the *marginal* velocity field to bend.

### 5.2 The spectrum of couplings

| Coupling           | `π(x_0, x_1)`                                                 | Cost per training step | Trajectory geometry / inference NFE |
|--------------------|---------------------------------------------------------------|------------------------|-------------------------------------|
| **Independent** (I-CFM) | `p_0(x_0) · q(x_1)` — sampled independently               | Trivial                | Crossing paths, curved marginal field, ~4–10 NFE |
| **Mini-batch OT** (OT-CFM)| Solve a Sinkhorn / Hungarian transport plan within each minibatch of size `B` | `O(B² log B)` per step | Paths run mostly parallel, ~2–4 NFE |
| **Continuous OT**  | `argmin_{π} E[‖x_1 − x_0‖²]` over all valid couplings        | Intractable in high dim | Provably straight in the limit; ~1 NFE in principle |
| **Reflow-induced** | Run a *trained* `v_θ` to pair `x_0 → ODE(v_θ, x_0) → x_1^pred`; retrain on those pairs | One full retraining round per reflow | Straightens iteratively — the Rectified Flow story (Section 6) |

The bottom three rows all attempt to buy back **trajectory
straightness**. The intuition: if noise is matched to *nearby* data,
then nearby `x_0`'s get traced to nearby `x_1`'s and the conditional
paths don't cross. No crossings ⇒ the marginal velocity field doesn't
have to curl ⇒ Euler with fewer steps remains accurate.

Independent coupling makes *no attempt* at matching. A noise sample near
the origin is paired with a data point drawn uniformly — which in
expectation is far away — so different conditional paths sweep wide
trajectories that overlap each other's territory. The marginal field
has to curl to be consistent with the crossings, and that curl is the
quantity that forces Euler to take small steps at inference.

### 5.3 Why I-CFM is still the working default

If OT couplings give straighter trajectories, why does FlowCast — and
most CFM applications in practice — use independent coupling?

1. **Mini-batch OT is a biased estimator of true OT.** The bias depends
   on batch size and data dimension; on high-resolution spatial fields
   the residual bias often eats the straightening benefit. The original
   I-CFM paper (Tong 2024) reports OT-CFM beating I-CFM on
   low-dimensional benchmarks but with shrinking gaps as dimension
   grows.

2. **Conditioning makes OT recomputation expensive.** For *conditional*
   generation — the regime FlowCast and FormosaFlow live in — the
   optimal coupling really depends on the conditioning bundle `c`,
   because the data distribution `q(x_1 | c)` shifts as `c` changes.
   Solving a per-batch, per-`c` OT problem is rarely affordable, and
   ignoring `c` in the transport plan gives the wrong coupling.

3. **Reflow gives most of the straightening for free.** Rectified Flow
   (Section 6) trains I-CFM once, then reuses the trained model to
   define a *learned* coupling, then retrains. No transport solver, no
   minibatch matching — and one or two reflow rounds typically suffice
   to recover most of the OT-CFM straightness benefit.

### 5.4 What "independent coupling specialization" means in one sentence

> Independent CFM is the corner of CFM design space where you **stop
> choosing**: independent endpoints, linear interpolation between them,
> tiny isotropic jitter. Every other CFM variant adds one of those
> choices back — an OT solve, a reflow round, a smarter path — to buy
> trajectory straightness at extra training-time cost.

This framing also clarifies what the **"Independent" in I-CFM is
*not***: it's not a claim about training-time independence between
samples, and it's not about path independence — it specifically means
the coupling `π(x_0, x_1)` factorizes as a product of its marginals.
Every richer CFM variant departs from that factorization in some way.

---

## 6. Rectified Flow

Rectified Flow (Liu et al. 2023) was developed in parallel to CFM and arrives
at almost the same training objective from a different angle.

### 6.1 The 1-rectified flow

Given coupled samples `(x_0, x_1)` (in the basic case, independent like
I-CFM), define the straight-line interpolant

```
x_t = (1 − t) x_0 + t x_1,    t ∈ [0, 1]
```

and learn a velocity field `v_θ` that matches the constant target `x_1 − x_0`
at every point:

```
L = E_{t, x_0, x_1} || v_θ(x_t, t) − (x_1 − x_0) ||²
```

This is **identical to I-CFM with `σ = 0`**. So at the training-objective
level, 1-rectified flow ≡ I-CFM with no path-thickening.

### 6.2 The reflow step (the actual rectified-flow contribution)

The interesting bit is what happens *after* training. The trained model
induces a *learned* coupling: starting from `x_0`, running the ODE forward
gives some `x_1^pred = ODE(v_θ, x_0)`. The **reflow** step retrains a *new*
velocity field using these learned pairs:

```
(x_0, x_1^pred)  →  retrain v_θ' on these straight-line interpolants
```

Each reflow iteration provably **straightens** the trajectories — in the
limit, paths become geodesically straight lines, and one Euler step suffices
to sample. The "rectified" in rectified flow refers to this straightening:

| Stage          | Trajectories look like | Min. ODE steps for good samples |
|----------------|------------------------|----------------------------------|
| 1-rectified    | curved (typical CFM)   | ~4–10                            |
| 2-rectified    | mostly straight        | ~2–4                             |
| k-rectified    | nearly straight        | 1 (with quality loss)            |
| + distillation | exactly straight       | 1                                |

So **rectified flow** = "straight-line flow matching" + iterative reflow +
optional distillation to a 1-step model.

### 6.3 Why FlowCast and most CFM papers stop at 1-rectified

Reflow is expensive (a whole retraining round per iteration) and the FlowCast
ablation (Table 9) shows that I-CFM with `σ = 0.01` already saturates quality
at ~10 Euler steps and is "almost optimal" at 1 step. So they skip reflow
and just use the base CFM training plus a fixed Euler sampler.

---

## 7. How CFM, I-CFM, and Rectified Flow relate

```
                     Flow Matching (Lipman 2023)
                     ─ general conditional Gaussian path
                     ─ choose μ_t, σ_t freely
                              │
              ┌───────────────┴─────────────────┐
              │                                 │
  ┌────────────────────────┐        ┌────────────────────────┐
  │ Diffusion-like paths   │        │ Straight-line paths    │
  │ (VE / VP schedules)    │        │ μ_t = (1-t)x_0 + t x_1 │
  │ Recover EDM, DDPM      │        │                        │
  └────────────────────────┘        └──────────┬─────────────┘
                                               │
                                ┌──────────────┴──────────────┐
                                │                             │
                  ┌─────────────────────────┐   ┌────────────────────────┐
                  │ I-CFM (Tong 2024)        │  │ Rectified Flow         │
                  │ Independent (x_0, x_1)   │  │ (Liu 2023)             │
                  │ σ > 0 path thickening    │  │ σ = 0                  │
                  │ One-shot training        │  │ Optional reflow rounds │
                  └─────────────┬───────────┘   └─────────────┬──────────┘
                                │                             │
                                └─────────────┬───────────────┘
                                              ▼
                                  Identical training step
                                  when σ = 0 and no reflow.
                                  Differ in (a) path width,
                                  (b) post-training procedure.
```

- **CFM** is the umbrella framework: pick any Gaussian conditional path.
- **I-CFM** is the straight-line, independent-coupling, `σ ≥ 0` instance —
  what FlowCast uses.
- **Rectified Flow** is the straight-line, `σ = 0` instance, plus the reflow
  refinement procedure.

If you only ever do one training round and never reflow, **I-CFM and
1-rectified flow are the same algorithm modulo the `σε` jitter**.

---

## 8. Practical implications for this repo

1. **Why FlowCast is fast to train compared to EDM.** Both are
   simulation-free (no ODE during training), but the CFM loss has a constant
   target velocity `x_1 − x_0` and no per-`σ` weighting schedule, whereas EDM
   needs the `(σ² + σ_data²) / (σ · σ_data)²` weighting and a log-normal `σ`
   sampler. See [stormcast/utils/flowcast_loss.py](stormcast/utils/flowcast_loss.py)
   vs the EDM loss in physicsnemo.

2. **Why FlowCast inference is fast.** Straight-line paths are easy for the
   Euler integrator. The paper shows good samples at 1–10 NFE; in this repo
   the default is 10 (config: `inference.num_steps=10` in
   [stormcast/config/flowcast_inference.yaml](stormcast/config/flowcast_inference.yaml)).

3. **Where you could go next if you want fewer NFE.** Two complementary moves:
   - Add **reflow**: retrain a second FlowCast on `(x_0, ODE(v_θ, x_0))`
     pairs. Conceptually clean, costs another training run.
   - Replace Euler with **Midpoint** (already supported via
     `flow_solver=midpoint` in [stormcast/utils/nn.py:163](stormcast/utils/nn.py#L163)).
     Doubles NFE per step but halves the step count for similar quality.

4. **`σ_path` (path thickening) is not the same as `sigma_data`.** In the
   loss, `sigma_data` standardizes the *target residual* into a unit-variance
   space; `sigma_path` is the I-CFM jitter inside the working space and is a
   small regularizer (FlowCast paper: 0.01). They live in different units
   and serve different purposes.

---

## 9. One-line summaries

- **CFM:** "Learn the velocity field by regressing against a tractable
  conditional vector field; the marginal one comes out for free in
  expectation."
- **I-CFM:** "Use a straight line between independent `(x_0, x_1)` with a
  tiny Gaussian jitter; the target velocity is just `x_1 − x_0`."
- **Rectified Flow:** "Same straight-line objective, then iteratively
  retrain on the model's own learned coupling so trajectories become literal
  straight lines you can integrate in one step."
- **FlowCast:** "I-CFM with `σ = 0.01`, conditioned on past frames (or, in
  our setup, the regression mean and the previous state), Euler with 10
  steps."
