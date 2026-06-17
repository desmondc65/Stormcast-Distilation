# Gated-Spectral MeanFlow (GS-MeanFlow)

A novel average-velocity residual head for the Taiwan RWRF StormCast stack,
designed to beat EDM diffusion, FlowCast (CFM) **and** vanilla MeanFlow on the
four high-res hourly variables (`t2m, u10, v10, qpepre`) while keeping
MeanFlow's 1–2 NFE sampling. This directory is the **MVP** (two of the three
proposed pillars); the third (spectral coarse-to-fine + climatological base
prior) is left as a documented extension.

The MVP is a strict superset of `MeanFlowPrecond`: with the gate disabled and
group-decoupling off it reproduces MeanFlow exactly, so every gain is
attributable to an added pillar.

---

## Why GS-MeanFlow should win (grounded in this repo's data)

Two measured pathologies in the existing results, not generic priors:

1. **A single shared objective can't win all four channels.** The qpw ablation
   (CLAUDE.md §6.5) shows winds prefer `qpw=1.4` (u10 0.273, v10 0.205) while
   t2m and qpepre prefer `qpw=2.0` (qpepre 0.385). One scalar weight on one
   shared velocity field forces a compromise; worse, MeanFlow's adaptive weight
   `w = 1/(mse+ε)^p` is computed per-sample over **all** channels, so a sample
   with hard qpepre down-weights its own easy wind channels. → **Pillar 1**.

2. **qpepre is not a continuous field.** It is a point mass at zero (dry) + a
   heavy right tail (wet). A continuous transport from `N(0,I)` cannot emit
   exact zeros, so it smears the dry/wet boundary — the documented "always
   slightly wet" failure (CLAUDE.md §8). → **Pillar 2**.

## The two MVP pillars

| Pillar | What | Where | Targets |
|---|---|---|---|
| **1. Group-decoupled adaptive weighting** | Smooth channels (t2m,u10,v10) and the precip channel (qpepre) each get their **own** per-sample adaptive weight, so they stop fighting over a shared `w`. | `gsmeanflow_loss.py` | u10/v10/t2m RMSE (removes interference) |
| **2. Occurrence (hurdle) gate** | A tiny deterministic U-Net predicts qpepre wet/dry from the conditioning `c`; focal-BCE trained. At inference, confidently-dry pixels are forced to **exact zero** precip. | `gsmeanflow_precond.py` (`_GateUNet`), composed in `nn_gsmeanflow.apply_occurrence_gate` | qpepre CSI / bias / RMSE ("always slightly wet") |

Everything else — the average-velocity matching (MeanFlow identity via JVP),
the per-channel β weights, and the qpepre log-PSD spectral regularizer — is
carried over unchanged from `MeanFlowLoss`/`MeanFlowPrecond`.

The gate operates on the conditioning only (never the noisy flow state), so it
is a clean occurrence model and costs one forward of a network ~50–100× smaller
than the SongUNet flow. Sampling stays at 1–2 NFE.

## Files

```
gated-spectral_meanflow/
├── gsmeanflow_precond.py   # GSMeanFlowPrecond: SongUNet avg-velocity flow + _GateUNet
├── gsmeanflow_loss.py      # GSMeanFlowLoss: group-decoupled MeanFlow + focal gate + log-PSD
├── nn_gsmeanflow.py        # factory, gated avg-velocity sampler, qpepre standardized levels
├── trainer_gsmeanflow.py   # training loop (gate target = state[1], gate-aware validation)
├── train_gsmeanflow.py     # Hydra entry (reuses ../config + ../utils + ../datasets)
├── inference_gsmeanflow.py # autoregressive rollout with the hurdle gate
├── train_gsmeanflow.sh     # torchrun launcher
├── docs/
│   ├── methodology.md          # full method reference (equations, pillars, ablations)
│   ├── make_method_figures.py  # regenerates the figures below (viridis thesis style)
│   └── figures/gsmf_*.png      # architecture / training / sampling / pillar diagrams
└── README.md
```

Method deep-dive with diagrams: [docs/methodology.md](docs/methodology.md).

Hydra configs live in the shared tree (so they sit beside every other method):
`config/gsmeanflow.yaml`, `config/model/gsmeanflow.yaml`,
`config/training/gsmeanflow.yaml`, `config/gsmeanflow_inference.yaml`,
`config/inference/gsmeanflow.yaml`.

## How to run

Launch from `stormcast/` (the entry scripts add the repo root + this dir to
`sys.path` and point Hydra at `../config`):

```bash
# Train (single process)
python gated-spectral_meanflow/train_gsmeanflow.py \
    model.regression_weights=/path/to/regression.mdlus \
    training.experiment_name=my_gsmeanflow_run

# Multi-GPU
bash gated-spectral_meanflow/train_gsmeanflow.sh

# Inference (uses the EMA shadow, same rule as FlowCast/MeanFlow)
python gated-spectral_meanflow/inference_gsmeanflow.py \
    inference.regression_checkpoint=/path/to/regression.mdlus \
    inference.gsmeanflow_ema_path=/path/to/run/ema_state.pt
```

Auto-resume: re-run the same command; the loop restores
`checkpoints_gsmeanflow/checkpoint_latest.pt` and `ema_state.pt`. **Use the EMA
weights for inference**, not the online student (same gotcha as FlowCast).

### Key knobs (Hydra overrides)

- `training.gate_weight` — strength of the occurrence term (0 disables the gate → pure decoupled MeanFlow).
- `training.gate_focal_alpha` / `training.gate_focal_gamma` — focal-BCE balance for sparse wet pixels.
- `training.gate_threshold_mm` — physical precip threshold for "wet" (default 0.1 mm/h, matches the CSI grid).
- `inference.gsmeanflow.gate_threshold` — wet-probability decision boundary at sampling time.
- `inference.gsmeanflow.apply_gate=false` — ablate the hurdle at inference without retraining.
- `model.gate_base_channels` — gate capacity.

## Ablation ladder (each isolates one pillar)

| Step | Config | Expected mover |
|---|---|---|
| 0 | vanilla MeanFlow (existing `train_meanflow.py`) | baseline |
| 1 | GS, `gate_weight=0` (Pillar 1 only) | u10/v10/t2m RMSE ↓ (no precip interference) |
| 2 | GS, full (Pillars 1+2) | + qpepre CSI/bias ↑, qpepre RMSE ↓ |
| 2b | GS, full, `inference.gsmeanflow.apply_gate=false` | isolates the gate's *inference-time* contribution |

Report each on the §7 metric suite (RMSE/MAE, CSI/FSS/bias at 0.1/1/5/10/20
mm/h, radial log-PSD) at matched sample budget. Add a `cleaned_gsmeanflow_nfe{1,2}`
row to the main comparison harness for the headline 4-way table.

## Notes / caveats

- **Validation RMSE is in standardized space** (like the existing CLAUDE.md
  headline numbers). The gate is applied to the standardized field before
  scoring, and dry pixels of both prediction and target sit at the same
  standardized 0-mm/h value, so correct dry predictions reduce RMSE directly.
- **Encoding compatibility:** the gate's wet threshold and dry value are
  computed through the loader's exact `log1p`+standardize pipeline
  (`qpepre_standardized_levels`), so this works on both the log1p cleaned
  dataset and a legacy raw-mm/h dataset (`qpepre_log1p=false`).
- **DDP correctness:** the gradient-carrying forward returns velocity **and**
  gate logits in one call, so both heads' params are used every step (no
  `find_unused_parameters`); the JVP target pass uses the flow-only path.
- **Risk:** decoupled heads optimize marginals; multivariate calibration
  (multivariate CRPS / rank histograms) should be checked, as it can regress
  even while per-channel RMSE improves. The shared SongUNet backbone mitigates
  this (cross-channel coupling is preserved through the flow).

## Pillar 3 (not in the MVP)

Spectral coarse-to-fine transport + a climatological-PSD base prior for qpepre,
reusing the MeanFlow `[r,t]` interval as a scale schedule (the band projection
is linear so the MeanFlow identity holds per band). Add only if the MVP clears
FlowCast on qpepre.
