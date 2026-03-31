# StormCast Distillation: Theory and Implementation

This document provides detailed theory and step-by-step implementation notes for the two distillation methods applied to the StormCast weather diffusion model: **Progressive Distillation** (Salimans & Ho, 2022) and **Consistency Distillation** (Song et al., 2023).

---

## Table of Contents

1. [Background: StormCast and EDM](#1-background-stormcast-and-edm)
2. [Progressive Distillation](#2-progressive-distillation)
   - [Theory](#21-theory)
   - [Algorithm](#22-algorithm)
   - [Implementation](#23-implementation)
   - [How to Run](#24-how-to-run)
3. [Consistency Distillation](#3-consistency-distillation)
   - [Theory](#31-theory)
   - [Algorithm](#32-algorithm)
   - [Implementation](#33-implementation)
   - [How to Run](#34-how-to-run)
4. [Shared Components](#4-shared-components)
5. [Output Structure and Metrics](#5-output-structure-and-metrics)
6. [Configuration Reference](#6-configuration-reference)

---

## 1. Background: StormCast and EDM

StormCast is a conditional diffusion model for high-resolution weather downscaling. It follows a two-stage architecture:

1. **Regression model** (`StormCastUNet`): A deterministic UNet that produces a mean prediction from coarse-resolution background inputs and the previous high-resolution state. This provides a strong prior for the diffusion model.

2. **Diffusion model** (`EDMPrecond`): Trained on the *residual* between the regression output and the ground truth. At inference, the regression output is added back to the diffusion sample.

The diffusion model follows the **EDM (Elucidated Diffusion Model)** framework (Karras et al., 2022), which uses a probability flow ODE (PF-ODE):

```
dx/dt = (x - D(x, t)) / t
```

where `D(x, t)` is the learned denoiser and `t` is the noise level (sigma). Sampling requires integrating this ODE from `t = sigma_max` down to `t = sigma_min`, which typically takes 18–256 Euler steps to achieve high quality — the computational bottleneck at inference.

Both distillation methods aim to reduce this step count while preserving sample quality.

### Karras Noise Schedule

Both methods use the Karras et al. noise schedule, which provides geometrically spaced noise levels:

```
sigma[i] = (sigma_max^(1/rho) + (i/N) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
```

Default parameters:
- `sigma_min = 0.002`
- `sigma_max = 80.0`
- `rho = 7.0`
- `sigma_data = 0.5`

**Implementation**: [`physicsnemo/experimental/metrics/diffusion/progressive_distillation_loss.py:99`](physicsnemo/experimental/metrics/diffusion/progressive_distillation_loss.py#L99) (`get_schedule`), [`physicsnemo/experimental/metrics/diffusion/consistency_loss.py:103`](physicsnemo/experimental/metrics/diffusion/consistency_loss.py#L103) (`get_discretization`)

### Model Architecture

All models (diffusion, progressive student, consistency) use the same `SongUNet` backbone with:
- `channel_mult = [1, 2, 2, 2, 2]`
- Configurable attention resolutions (`attn_resolutions = []` by default)
- Optional additive positional spatial embedding

The `EDMPrecond` wrapper applies EDM preconditioning (c_skip, c_out, c_in, c_noise scaling), while `ConsistencyPrecond` applies the consistency model boundary condition: `f(x, sigma_min) = x`.

**Implementation**: [`stormcast/utils/nn.py:25`](stormcast/utils/nn.py#L25) (`get_preconditioned_architecture`)

### Conditioning

The diffusion model is conditioned on a concatenation of:
- `state`: Previous high-resolution atmospheric state (`t-1`)
- `regression`: Output of the regression network (residual target mode)
- `invariant`: Static fields (land-sea mask, orography)

The regression network is itself conditioned on `[state, background, invariant]`.

**Implementation**: [`stormcast/utils/nn.py:80`](stormcast/utils/nn.py#L80) (`build_network_condition_and_target`)

---

## 2. Progressive Distillation

### 2.1 Theory

**Reference**: Salimans, T. and Ho, J., 2022. *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR 2022.

Progressive distillation is an iterative procedure that reduces the number of sampling steps by a factor of 2 at each phase. The key insight is:

> **Two consecutive Euler steps of a teacher can be approximated by one Euler step of a student.**

Formally, suppose the teacher uses a schedule of 2N steps with noise levels `{sigma_0, sigma_1, ..., sigma_{2N}}`. The student operates on the even-indexed subset `{sigma_0, sigma_2, ..., sigma_{2N}}`, i.e., N steps. The student is trained so that starting from any noisy sample `x` at `sigma_{2i}`, a single student Euler step to `sigma_{2i+2}` matches the outcome of two teacher Euler steps through `sigma_{2i+1}`.

This process is applied repeatedly:
- Phase 0: Teacher uses 2·`initial_steps` schedule; student learns N = `initial_steps` steps
- Phase 1: Teacher = Phase 0 student; student learns N/2 steps
- Phase k: Teacher = Phase k-1 student; student learns N/2^k steps
- Stop when student reaches `target_num_steps`

The total number of phases is `ceil(log2(initial_steps / target_steps))`.

#### Loss Function

For each training sample, a random student step index `i ∈ {0, ..., N-1}` is drawn. The three relevant noise levels are:

```
sigma_start = schedule[2i]    (student starts here)
sigma_mid   = schedule[2i+1]  (teacher intermediate)
sigma_end   = schedule[2i+2]  (student ends here)
```

A noisy sample is created: `x = x_clean + sigma_start * eps`, where `eps ~ N(0, I)`.

The teacher executes two Euler steps (frozen, no gradient):
```
denoised_1 = D_teacher(x, sigma_start)
d_1 = (x - denoised_1) / sigma_start
x_mid = x + (sigma_mid - sigma_start) * d_1

denoised_2 = D_teacher(x_mid, sigma_mid)
d_2 = (x_mid - denoised_2) / sigma_mid
x_target = x_mid + (sigma_end - sigma_mid) * d_2
```

The student executes one Euler step (gradient flows):
```
denoised_s = D_student(x, sigma_start)
d_s = (x - denoised_s) / sigma_start
x_student = x + (sigma_end - sigma_start) * d_s
```

The loss is:
```
L = MSE(x_student, x_target.detach())
```

Optionally, EDM-style per-sample weighting can be applied:
```
w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
L = w(sigma_start) * MSE(x_student, x_target)
```

**Note**: The loss is in *trajectory space* (noisy image space), not in clean image space. This is important because it avoids requiring the model to denoise in a single step from any noise level — it only needs to match the trajectory of the teacher.

### 2.2 Algorithm

```
Algorithm: Progressive Distillation

Input: Pre-trained EDM teacher T_0, initial_steps N_0, target_steps N_T, steps_per_phase K

num_phases = ceil(log2(N_0 / N_T))

For phase = 0, 1, ..., num_phases-1:
    N_teacher = N_0 / 2^phase         # Teacher operates at this many steps
    N_student = N_0 / 2^(phase+1)     # Student operates at half this
    N_student = max(N_student, N_T)

    teacher_schedule = Karras(2 * N_student)   # 2N+1 sigma values

    Initialize student S from teacher T_phase weights

    For training_step = 1, ..., K:
        Sample batch (x_clean, condition)
        Sample i ~ Uniform{0, ..., N_student-1}
        sigma_start, sigma_mid, sigma_end = teacher_schedule[2i], [2i+1], [2i+2]

        x_noisy = x_clean + sigma_start * N(0, I)

        # Teacher trajectory (no grad)
        x_target = Euler2(T_phase, x_noisy, sigma_start -> sigma_mid -> sigma_end)

        # Student trajectory (grad)
        x_student = Euler1(S, x_noisy, sigma_start -> sigma_end)

        loss = MSE(x_student, x_target)  [+ optional EDM weighting]
        loss.backward(); optimizer.step()

    T_{phase+1} = copy.deepcopy(S)    # Promote student to teacher

    If N_student <= N_T: break

Output: Final student S (samples in N_T steps via Euler integration)
```

### 2.3 Implementation

#### Entry Point

[`stormcast/train_progressive.py`](stormcast/train_progressive.py) — Hydra entry point. Initializes distributed training, W&B, and calls `progressive_distillation_loop`.

#### Training Loop

[`stormcast/utils/trainer_progressive.py`](stormcast/utils/trainer_progressive.py) — `progressive_distillation_loop(cfg)`

Key implementation details:

**Phase setup** (line 246–283):
```python
for phase in range(num_phases):
    N_student = initial_steps // (2 ** (phase + 1))
    N_student = max(N_student, target_steps)
    loss_fn.set_num_steps(N_student)

    # Student initialized from teacher weights
    student = get_preconditioned_architecture(name="diffusion", **arch_kwargs)
    student.load_state_dict(teacher.state_dict())
```

**Training step** (line 317–380):
```python
while phase_step < steps_per_phase:
    optimizer.zero_grad(set_to_none=True)
    for _ in range(num_accumulation_rounds):    # gradient accumulation
        batch = next(dataset_iterator)
        condition, target, reg_out = build_network_condition_and_target(...)
        loss = loss_fn(student=ddp, teacher=teacher, images=target, condition=condition)
        (loss.sum() / len(state_channels)).backward()
    clip_grad_norm_(student.parameters(), cfg.training.clip_grad_norm)
    optimizer.step()
```

**Phase promotion** (line 531–536):
```python
teacher = copy.deepcopy(student)
teacher.eval().requires_grad_(False)
```

**Validation** uses `diffusion_model_forward` with `N_student` steps and the Euler solver, then computes RMSE, MAE, and power spectra.

#### Loss Function

[`physicsnemo/experimental/metrics/diffusion/progressive_distillation_loss.py`](physicsnemo/experimental/metrics/diffusion/progressive_distillation_loss.py) — `ProgressiveDistillationLoss`

Key method: `__call__(student, teacher, images, condition)` (line 206)

- Builds `2*N`-step teacher schedule
- Samples random student step indices `i` per batch element
- Calls `_teacher_two_steps` (no grad) and `_student_one_step` (with grad)
- Returns unreduced loss tensor `(B, C, H, W)`

`set_num_steps(N)` must be called at the start of each phase to update `N`.

#### Resume Logic

[`stormcast/utils/trainer_progressive.py:556`](stormcast/utils/trainer_progressive.py#L556) — `_detect_resume_state`

Scans phase directories backwards for `student_final.mdlus` (phase complete) or checkpoint `.pt` files (phase partially complete). Returns `(resume_phase, resume_step)`.

### 2.4 How to Run

#### Prerequisites

- Pre-trained EDM checkpoint: `path/to/edm_teacher.mdlus`
- Pre-trained regression checkpoint: `path/to/regression.mdlus`
- Dataset configured in [`stormcast/config/dataset/`](stormcast/config/dataset/)

#### Single GPU

```bash
cd stormcast/

python train_progressive.py \
    model.teacher_weights=/path/to/edm_teacher.mdlus \
    model.regression_weights=/path/to/regression.mdlus \
    training.experiment_name=my_progressive_run
```

#### Multi-GPU (DDP)

```bash
cd stormcast/

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    train_progressive.py \
    model.teacher_weights=/path/to/edm_teacher.mdlus \
    model.regression_weights=/path/to/regression.mdlus \
    training.experiment_name=my_progressive_run_4gpu
```

#### Common Config Overrides

```bash
# Change phase schedule (default: 128 -> 4 steps, 5 phases)
python train_progressive.py \
    training.initial_num_steps=256 \
    training.target_num_steps=2 \
    training.steps_per_phase=100000

# Use EDM loss weighting instead of uniform
python train_progressive.py \
    training.loss_weighting=edm

# Enable W&B logging
python train_progressive.py \
    training.log_to_wandb=True

# Adjust batch size / learning rate
python train_progressive.py \
    training.batch_size=128 \
    training.lr=5e-5

# Save NetCDF output during validation
python train_progressive.py \
    training.output_nc=True \
    training.output_nc_freq=10
```

#### Resume

Training auto-resumes from the latest checkpoint. Just re-run the same command:

```bash
python train_progressive.py \
    model.teacher_weights=/path/to/edm_teacher.mdlus \
    model.regression_weights=/path/to/regression.mdlus \
    training.experiment_name=my_progressive_run
```

The `_detect_resume_state` function finds the latest phase and step automatically.

#### Output

```
StormCast_progressive/my_progressive_run/0/
├── phase_0/                         # Teacher: 256 steps -> Student: 128 steps
│   ├── checkpoints/checkpoint_*.pt  # Optimizer + model state
│   ├── student_final.mdlus          # Saved student (used as next phase teacher)
│   ├── train_loss.csv
│   ├── valid_loss.csv
│   ├── rmse_{field}.csv
│   ├── mae_{field}.csv
│   ├── ps1d_{field}.csv
│   └── images/{field}/*.png
├── phase_1/                         # Teacher: 128 steps -> Student: 64 steps
│   └── ...
└── phase_N/                         # Final student at target_num_steps
    └── ...
```

The final distilled model is `phase_N/student_final.mdlus`. Use `progressive_distilled_forward()` in [`stormcast/utils/nn.py:180`](stormcast/utils/nn.py#L180) for inference.

---

## 3. Consistency Distillation

### 3.1 Theory

**Reference**: Song, Y., Dhariwal, P., Chen, M. and Sutskever, I., 2023. *Consistency Models*. ICML 2023.

Consistency distillation trains a *consistency model* — a function `f(x, sigma)` that maps any noisy point `(x, sigma)` on an ODE trajectory directly to the trajectory's endpoint at `sigma_min`. This enables **1-step generation** (or few-step with iterative refinement).

The defining property is the *consistency condition*:
```
f(x_t, t) = f(x_t', t')   for all t, t' on the same ODE trajectory
```

At the boundary: `f(x, sigma_min) = x` (the model is the identity at minimum noise).

#### Training Objective

The consistency model is trained via *distillation* from a pre-trained EDM teacher. For two consecutive points on the ODE trajectory at noise levels `t_{n+1} > t_n`:

1. Create noisy sample: `x_{n+1} = x_clean + t_{n+1} * eps`
2. Use the teacher to take one Euler step: `x_hat_n = Euler(x_{n+1}, t_{n+1} -> t_n)`
3. The consistency condition requires: `f(x_{n+1}, t_{n+1}) ≈ f(x_hat_n, t_n)`

The loss is:
```
L_CD = d(f_theta(x_{n+1}, t_{n+1}),  f_theta_ema(x_hat_n, t_n)) / N(k)
```

where:
- `f_theta` is the online student (gradient flows)
- `f_theta_ema` is an EMA target network (stop gradient — provides stable targets)
- `d(·,·)` is a distance metric (MSE or pseudo-Huber loss)
- `N(k)` is the adaptive number of discretization steps at training step `k`

The division by `N(k)` normalizes the loss as the discretization grows finer.

#### EMA Target Network

The EMA target `f_theta_ema` provides stable training targets (analogous to the target network in DQN). It is updated after each optimizer step:

```
theta_ema <- mu_k * theta_ema + (1 - mu_k) * theta
```

The decay `mu_k` adapts with the number of discretization steps per Song et al. Eq. 14:

```
mu_k = mu_0^(N_0 / N_k)
```

As `N_k` grows, `mu_k -> 1`, meaning the EMA target updates more slowly when the discretization is finer (more stable targets needed).

**Implementation**: [`stormcast/utils/ema.py`](stormcast/utils/ema.py) — `ExponentialMovingAverage` and `ema_decay_schedule`

#### Adaptive Discretization Schedule N(k)

The number of discretization steps grows from `N_0` to `N_total` over training using a sqrt schedule:

```
N(k) = ceil(sqrt((k/K) * ((N_total+1)^2 - N_0^2) + N_0^2) - 1) + 1
```

where `K = total_train_steps`. This gradually increases the number of consistency pairs, making the training target progressively harder but also more accurate.

**Implementation**: [`physicsnemo/experimental/metrics/diffusion/consistency_loss.py:78`](physicsnemo/experimental/metrics/diffusion/consistency_loss.py#L78) (`get_num_steps`)

#### Pseudo-Huber Loss

The loss metric `d(·,·)` uses a pseudo-Huber loss (smooth L1) rather than MSE, following Song et al.:

```
d(a, b) = sqrt(||a - b||^2 + c^2) - c
```

where `c = huber_c = 0.00054` (default). This is more robust to outliers than MSE while remaining differentiable everywhere. Setting `huber_c = None` falls back to MSE.

**Implementation**: [`physicsnemo/experimental/metrics/diffusion/consistency_loss.py:230`](physicsnemo/experimental/metrics/diffusion/consistency_loss.py#L230)

#### Inference

At inference, a consistency model generates samples in **one step**:

```
x ~ N(0, sigma_max^2 * I)
sample = f(x, sigma_max)
```

The `ConsistencyPrecond` wrapper enforces the boundary condition: the model output at `sigma = sigma_min` is the identity (input passed through unchanged), so no special handling is needed at the clean end.

**Implementation**: [`stormcast/utils/nn.py:160`](stormcast/utils/nn.py#L160) (`consistency_model_forward`)

### 3.2 Algorithm

```
Algorithm: Consistency Distillation

Input: Pre-trained EDM teacher T (frozen), N_0, N_total, K = total_train_steps
       Initial EMA decay mu_0

Initialize student S (ConsistencyPrecond, weights from T)
Initialize EMA target S_ema = copy(S)
Initialize ExponentialMovingAverage tracker EMA

For training_step k = 0, 1, ..., K-1:
    # Adaptive discretization
    N_k = ceil(sqrt((k/K) * ((N_total+1)^2 - N_0^2) + N_0^2) - 1) + 1
    mu_k = mu_0^(N_0 / N_k)

    t_schedule = Karras(N_k)     # N_k+1 noise levels

    Sample batch (x_clean, condition)
    Sample n ~ Uniform{1, ..., N_k}
    t_{n+1} = t_schedule[n-1]    # noisier level
    t_n     = t_schedule[n]      # less noisy level

    x_{n+1} = x_clean + t_{n+1} * N(0, I)

    # Teacher Euler step (no grad)
    x_hat_n = Euler(T, x_{n+1}, t_{n+1} -> t_n)

    # Online student forward (grad)
    out_student = S(x_{n+1}, t_{n+1})

    # EMA target forward (no grad)
    out_target = S_ema(x_hat_n, t_n)

    # Pseudo-Huber loss
    loss = (sqrt(||out_student - out_target||^2 + c^2) - c) / N_k
    loss.backward(); optimizer.step()

    # Update EMA target
    EMA.update(S, decay=mu_k)
    EMA.apply_shadow(S_ema)

Output: S (1-step generation: sample = S(x ~ N(0, sigma_max^2 I), sigma_max))
```

### 3.3 Implementation

#### Entry Point

[`stormcast/train_consistency.py`](stormcast/train_consistency.py) — Hydra entry point. Initializes distributed training, W&B, and calls `consistency_training_loop`.

#### Training Loop

[`stormcast/utils/trainer_consistency.py`](stormcast/utils/trainer_consistency.py) — `consistency_training_loop(cfg)`

Key implementation details:

**Model setup**: Student is `ConsistencyPrecond` initialized from teacher weights. A separate `ema_net` (also `ConsistencyPrecond`) is kept as the EMA shadow copy, updated via `ExponentialMovingAverage`.

**Per-step EMA update**:
```python
N_k = loss_fn.get_num_steps(total_steps)
mu_k = ema_decay_schedule(cfg.training.N_0, N_k, cfg.training.ema_decay_init)
ema.update(student, decay=mu_k)
ema.apply_shadow(ema_net)
```

**Loss call**:
```python
loss = loss_fn(
    student=ddp,
    teacher=teacher,
    ema_student=ema_net,
    images=target,
    condition=condition,
    current_step=total_steps,
)
```

**Validation** uses `consistency_model_forward` (1-step: pure Gaussian noise → model forward), then computes RMSE, MAE, and power spectra.

**Checkpointing** saves both the student model state and the EMA shadow parameters:
```
checkpoints_consistency/checkpoint_*.pt    # Student + optimizer
ema_state.pt                               # EMA shadow parameters
```

#### Loss Function

[`physicsnemo/experimental/metrics/diffusion/consistency_loss.py`](physicsnemo/experimental/metrics/diffusion/consistency_loss.py) — `ConsistencyDistillationLoss`

- `get_num_steps(current_step)` → adaptive `N(k)` (line 78)
- `get_discretization(N, device)` → Karras schedule tensor (line 103)
- `euler_step(teacher, x, t_cur, t_next, condition)` → single teacher Euler step (line 129)
- `__call__(student, teacher, ema_student, images, condition, current_step)` → loss (line 165)

#### EMA

[`stormcast/utils/ema.py`](stormcast/utils/ema.py)

- `ExponentialMovingAverage`: stores shadow parameters; `update(model, decay)` does `lerp_`; `apply_shadow(model)` copies weights into target model
- `ema_decay_schedule(N_0, N_k, mu_0)` → `mu_0^(N_0 / N_k)` (line 92)

### 3.4 How to Run

#### Prerequisites

Same as progressive distillation:
- Pre-trained EDM checkpoint: `path/to/edm_teacher.mdlus`
- Pre-trained regression checkpoint: `path/to/regression.mdlus`
- Dataset configured in [`stormcast/config/dataset/`](stormcast/config/dataset/)

#### Single GPU

```bash
cd stormcast/

python train_consistency.py \
    model.teacher_weights=/path/to/edm_teacher.mdlus \
    model.regression_weights=/path/to/regression.mdlus \
    training.experiment_name=my_consistency_run
```

#### Multi-GPU (DDP)

```bash
cd stormcast/

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    train_consistency.py \
    model.teacher_weights=/path/to/edm_teacher.mdlus \
    model.regression_weights=/path/to/regression.mdlus \
    training.experiment_name=my_consistency_run_4gpu
```

#### Common Config Overrides

```bash
# Increase total training steps
python train_consistency.py \
    training.total_train_steps=600000

# Adjust adaptive schedule endpoints
python train_consistency.py \
    training.N_0=1 \
    training.N_total=200

# Change EMA decay base
python train_consistency.py \
    training.ema_decay_init=0.99

# Adjust pseudo-Huber constant (or disable it)
python train_consistency.py \
    training.huber_c=0.001     # larger -> closer to L1 for large errors
# training.huber_c=0.0        # disable -> MSE

# Enable W&B logging
python train_consistency.py \
    training.log_to_wandb=True

# Adjust learning rate / batch size
python train_consistency.py \
    training.lr=5e-5 \
    training.batch_size=128
```

#### Resume

Training auto-resumes from `checkpoints_consistency/checkpoint_latest.pt`. Just re-run the same command.

#### Output

```
StormCast_consistency/my_consistency_run/0/
├── checkpoints_consistency/
│   └── checkpoint_*.pt      # Student model + optimizer state
├── ema_state.pt              # EMA shadow parameters (the actual inference model)
├── train_loss.csv
├── valid_loss.csv
├── rmse_{field}.csv
├── mae_{field}.csv
├── ps1d_{field}.csv
└── images/{field}/*.png
```

The final consistency model for inference is the **EMA model** (`ema_state.pt`), not the online student. Load it via:

```python
from physicsnemo.models.diffusion import ConsistencyPrecond
from stormcast.utils.ema import ExponentialMovingAverage

student = ConsistencyPrecond(...)
ema = ExponentialMovingAverage(student)
ema.load_state_dict(torch.load("ema_state.pt"))
ema.apply_shadow(student)   # student now has EMA weights

# 1-step generation
from stormcast.utils.nn import consistency_model_forward
sample = consistency_model_forward(student, condition, shape, sigma_max=80.0)
```

---

## 4. Shared Components

### Model Factory

[`stormcast/utils/nn.py:25`](stormcast/utils/nn.py#L25) — `get_preconditioned_architecture(name, ...)`

| `name`          | Returns                | Notes                                        |
|-----------------|------------------------|----------------------------------------------|
| `"regression"`  | `StormCastUNet`        | Deterministic; `embedding_type="zero"`       |
| `"diffusion"`   | `EDMPrecond`           | Used for base EDM + progressive distillation |
| `"consistency"` | `ConsistencyPrecond`   | Adds boundary condition at `sigma_min`       |

All use `SongUNet` backbone with `channel_mult=[1,2,2,2,2]`.

### Conditioning Helper

[`stormcast/utils/nn.py:80`](stormcast/utils/nn.py#L80) — `build_network_condition_and_target(background, state, invariant_tensor, regression_net, condition_list, regression_condition_list)`

Returns `(condition, target, regression_output)` where:
- `condition` = concatenation of tensors in `condition_list`
- `target` = `state[1]` (next state) minus `regression_output` if `"regression"` in condition list
- `regression_output` = None if regression is not used

### Inference Functions

[`stormcast/utils/nn.py`](stormcast/utils/nn.py)

| Function | Description |
|---|---|
| `diffusion_model_forward(model, condition, shape, sampler_args)` | Standard EDM deterministic sampler |
| `progressive_distilled_forward(model, condition, shape, num_steps, ...)` | Euler sampler with reduced steps |
| `consistency_model_forward(model, condition, shape, sigma_max)` | 1-step: sample from `N(0, sigma_max^2 I)` |
| `regression_model_forward(model, state, background, invariant, ...)` | Direct regression prediction |

### Distributed Training

All training loops use `DistributedManager` from `physicsnemo.distributed`. The loop detects the number of GPUs automatically. Gradient accumulation is used to achieve effective batch size regardless of GPU count:

```
num_accumulation_rounds = batch_size / (batch_size_per_gpu * world_size)
```

Loss is `all_reduce`d across ranks before logging.

### Hydra Configuration

The config system uses [Hydra](https://hydra.cc/) with config groups composed at runtime. The top-level configs ([`stormcast/config/progressive.yaml`](stormcast/config/progressive.yaml), [`stormcast/config/consistency.yaml`](stormcast/config/consistency.yaml)) compose `model/`, `training/`, `dataset/`, and `sampler/` sub-configs.

Any config value can be overridden on the command line using dot-notation: `training.lr=1e-5`, `model.sigma_max=100.0`.

---

## 5. Output Structure and Metrics

### Training Logs (CSV)

| File | Content |
|---|---|
| `train_loss.csv` | `global_step, loss` — average loss per print interval |
| `valid_loss.csv` | `global_step, loss` — distillation loss on validation batch |
| `rmse_{field}.csv` | `global_step, rmse` — per-field RMSE (averaged over batch) |
| `mae_{field}.csv` | `global_step, mae` — per-field MAE |
| `ps1d_{field}.csv` | `global_step, k, Pk_gen, Pk_tar` — radially averaged 1D power spectra |

### Validation Images

For each field in `validation_plot_variables` (`t2m`, `u10`, `v10`, `qpepre` by default):
- `{step}_{i}_{field}.png` — Side-by-side generated vs. truth heatmap
- `{step}_{i}_{field}_spec.png` — 1D power spectrum comparison

### Checkpoints

**Progressive**:
- `phase_{n}/checkpoints/checkpoint_{step}.pt` — PyTorch checkpoint (model + optimizer)
- `phase_{n}/student_final.mdlus` — PhysicsNeMo model file (used as next phase teacher)

**Consistency**:
- `checkpoints_consistency/checkpoint_{step}.pt` — Student model + optimizer
- `ema_state.pt` — EMA shadow parameters (inference model)

### W&B Integration

Set `training.log_to_wandb=True` and optionally `training.wandb_mode=online|offline|disabled`.

Logged quantities include: `loss`, `valid_loss`, `lr`, `phase` / `N_student` (progressive), `N_k` / `mu_k` (consistency), per-field validation images and power spectrum plots.

---

## 6. Configuration Reference

### Progressive Distillation

**Top-level**: [`stormcast/config/progressive.yaml`](stormcast/config/progressive.yaml) — composes `model: progressive`, `training: progressive`, `dataset: era5_rwrf_qpepre`, `sampler: edm_deterministic`

#### Model Config ([`stormcast/config/model/progressive.yaml`](stormcast/config/model/progressive.yaml))

| Key | Default | Description |
|---|---|---|
| `model_name` | `progressive` | Model identifier |
| `regression_conditions` | `[state, background, invariant]` | Inputs to regression model |
| `diffusion_conditions` | `[state, regression, invariant]` | Inputs to diffusion/student model |
| `spatial_pos_embed` | `False` | Additive spatial positional embedding |
| `attn_resolutions` | `[]` | UNet stages to use self-attention |
| `regression_weights` | `""` | **Required**: path to regression checkpoint |
| `teacher_weights` | `""` | **Required**: path to EDM teacher checkpoint |
| `sigma_min` | `0.002` | Minimum noise level |
| `sigma_max` | `80.0` | Maximum noise level |
| `sigma_data` | `0.5` | Expected data standard deviation |

#### Training Config ([`stormcast/config/training/progressive.yaml`](stormcast/config/training/progressive.yaml))

| Key | Default | Description |
|---|---|---|
| `initial_num_steps` | `128` | Starting student step count (teacher uses 2×) |
| `target_num_steps` | `4` | Stop when student reaches this step count |
| `steps_per_phase` | `50000` | Optimizer steps per distillation phase |
| `rho` | `7.0` | Karras schedule exponent |
| `loss_weighting` | `uniform` | `uniform` or `edm` per-sample weighting |
| `batch_size` | `64` | Global batch size |
| `lr` | `1e-4` | Learning rate |
| `lr_rampup_steps` | `500` | Linear LR warmup steps |
| `clip_grad_norm` | `1.0` | Gradient clipping norm |
| `checkpoint_freq` | `5000` | Save checkpoint every N steps |
| `validation_freq` | `500` | Run validation every N steps |

### Consistency Distillation

**Top-level**: [`stormcast/config/consistency.yaml`](stormcast/config/consistency.yaml) — composes `model: consistency`, `training: consistency`, `dataset: era5_rwrf_qpepre`, `sampler: edm_deterministic`

#### Model Config ([`stormcast/config/model/consistency.yaml`](stormcast/config/model/consistency.yaml))

Same keys as progressive model config, but `model_name: consistency`. The `teacher_weights` and `regression_weights` fields must be set.

#### Training Config ([`stormcast/config/training/consistency.yaml`](stormcast/config/training/consistency.yaml))

| Key | Default | Description |
|---|---|---|
| `total_train_steps` | `400000` | Total optimizer steps |
| `N_0` | `2` | Initial discretization steps |
| `N_total` | `150` | Final discretization steps |
| `rho` | `7.0` | Karras schedule exponent |
| `huber_c` | `0.00054` | Pseudo-Huber constant (`None` for MSE) |
| `ema_decay_init` | `0.95` | Base EMA decay `mu_0` |
| `batch_size` | `64` | Global batch size |
| `lr` | `1e-4` | Learning rate |
| `lr_rampup_steps` | `500` | Linear LR warmup steps |
| `clip_grad_norm` | `1.0` | Gradient clipping norm |
| `checkpoint_freq` | `5000` | Save checkpoint every N steps |
| `validation_freq` | `500` | Run validation every N steps |

---

## Key Differences Between Methods

| Aspect | Progressive Distillation | Consistency Distillation |
|---|---|---|
| **Goal** | Reduce steps from N to N/2 per phase | Single-step generation |
| **Training phases** | Multiple (one per 2× reduction) | Single continuous training run |
| **Teacher** | Updated each phase (previous student) | Fixed EDM teacher throughout |
| **Target network** | Teacher is exact (no EMA) | EMA of online student |
| **Loss target** | Trajectory point (noisy space) | Denoised prediction (clean space) |
| **Loss metric** | MSE in trajectory space | Pseudo-Huber in prediction space |
| **Inference** | EDM Euler sampler, few steps | Direct 1-step: `f(x, sigma_max)` |
| **Model type** | `EDMPrecond` (same as teacher) | `ConsistencyPrecond` (different preconditioning) |
| **Schedule** | Fixed N during phase | Adaptive N(k), grows over training |
| **Inference file** | `phase_N/student_final.mdlus` | `ema_state.pt` (EMA shadow) |
