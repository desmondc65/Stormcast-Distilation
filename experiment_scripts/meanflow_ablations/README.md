# MeanFlow ablation suite

Controlled ablations of the StormCast **MeanFlow** student (average-velocity
flow matching on the regression residual `R_t = X_t − M_t`, Geng et al. 2025),
each changing **one** knob relative to the production recipe so the resulting
metrics are clean A/Bs.

Implementation under ablation:
[stormcast/utils/meanflow_loss.py](../../stormcast/utils/meanflow_loss.py),
[stormcast/utils/meanflow_precond.py](../../stormcast/utils/meanflow_precond.py),
[stormcast/utils/trainer_meanflow.py](../../stormcast/utils/trainer_meanflow.py),
launched via [stormcast/train_meanflow.py](../../stormcast/train_meanflow.py).

## Protocol (identical across every run)

| Setting | Value | Why |
|---|---|---|
| GPUs | 7 (`CUDA_VISIBLE_DEVICES=0..6`) | the zettabyte worker |
| `batch_size` | 112 (7 × 16/GPU) | must be a multiple of `world_size=7` |
| `total_train_steps` | 18000 | 18000 × 112 = **2,016,000 ≈ 2.0M samples** |
| `checkpoint_freq` | 18000 (== total) | **one** checkpoint, written at the final step |
| LR schedule | warmup 1000 → cosine → `min_lr` at 18000 | self-contained, fully decayed at the 2M budget |
| EMA | 0.999 (inference weights = `ema_state.pt`) | matches production |
| dataset | cleaned 192×96, `qpepre_log1p=true` | matches production |
| regression anchor | `StormCastUNet.0.8000.mdlus` (frozen) | matches production |

The 2M-sample budget matches the deployed headline leg
(`MeanFlowPrecond.0.20000.mdlus`, the repo's "~2M-sample budget") closely
enough for direct comparison while keeping each ablation cheap. All shared
config lives in [`_common.sh`](_common.sh); the per-ablation scripts are thin
wrappers that flip one knob.

**Only one checkpoint is saved per run** (`MeanFlowPrecond.0.18000.mdlus` +
`ema_state.pt`), at ~2M samples, by setting `checkpoint_freq == total_train_steps`.
Validation still runs every 500 steps for the loss / RMSE / PS1D CSVs and
heatmaps, but those write no weights. Trade-off: an interrupted run has no
resume point and restarts from scratch — lower `CHECKPOINT_FREQ` in `_common.sh`
if you want intermediate saves.

## The ablations

`00_baseline` is the reference; read every other run as a delta from it.

### Core — the questions most central to the method
| Script | Knob | Question |
|---|---|---|
| `00_baseline` | — | production recipe @ 2M; reference |
| `01_mf_ratio_0.00` | `mf_ratio=0.0` | **Does the MeanFlow identity help at all?** 0.0 = plain I-CFM (no average-velocity bootstrap) evaluated at few NFE. |
| `02_mf_ratio_0.50` | `mf_ratio=0.5` | more batch on the bootstrapped target |
| `03_mf_ratio_0.75` | `mf_ratio=0.75` | how far the ratio can be pushed |
| `04_mf_ratio_1.00` | `mf_ratio=1.0` | no I-CFM anchor — documents the unstable extreme |
| `05_adaptive_off` | `adaptive_p=0.0` | is the adaptive `1/(mse+eps)^p` weight worth it on heavy-tailed qpepre? |
| `07_spectral_off` | `spectral_weight=0.0` | does the qpepre log-PSD term keep the spectrum calibrated? |
| `09_chanw_uniform` | `channel_weights=[1,1,1,1]` | does the 2× qpepre boost actually help precip? |

### Extended — bracketing / secondary axes
| Script | Knob | Question |
|---|---|---|
| `06_adaptive_p0.5` | `adaptive_p=0.5` | turns the adaptive on/off into a trend |
| `08_spectral_strong` | `spectral_weight=0.5` | other side of the spectral knob (brackets 0.1) |
| `10_chanw_qpepre_strong` | `channel_weights=[1,1,1,5]` | precip-skill vs smooth-channel-RMSE trade-off |
| `11_ema_0.9999` | `ema_decay=0.9999` | slower EMA → deployed weights smoother or lagging? |
| `12_no_pos_embed` | `model.spatial_pos_embed=False` | how much skill comes from the learned spatial prior on this fixed domain? |
| `13_attn` | `model.attn_resolutions=[24]` | does global self-attention help organize convective structure? |

Channel order is `u10, v10, t2m, qpepre`, so `channel_weights[3]` is qpepre.

## Running

```bash
cd experiment_scripts/meanflow_ablations

./run_all.sh            # core subset (recommended first pass), sequential
./run_all.sh all        # everything
./run_all.sh 01 13      # only scripts starting 01_/13_
bash 00_baseline.sh     # a single ablation
```

Each run is launched with `torchrun` on all 7 GPUs, so they run one at a time.
`run_all.sh` skips any ablation whose final checkpoint already exists, so it is
safe to re-run. Per-run logs are mirrored to `logs/<exp>_<timestamp>.log`.

## What to report (2022 validation year)

The trainer already writes `train_loss.csv`, `valid_loss.csv`,
`rmse_<field>.csv`, `mae_<field>.csv`, `ps1d_<field>.csv` and heatmaps per run.
For the thesis the decisive comparisons are:

- **Few-step quality vs `mf_ratio`** — re-sample each checkpoint at 1 and 2 NFE
  (the validation default is 2). The MeanFlow story lives in the **1-NFE** gap
  between `mf_ratio=0` and the `>0` runs.
- **qpepre precip skill** — CSI / FSS / frequency-of-exceedance at
  0.1/1/5/10/20 mm/h, plus the radial log-PSD — for the spectral and
  channel-weight ablations (RMSE alone hides over-smoothing).
- **qpepre mode collapse** — wet/dry pixel fraction and 95/99th percentiles,
  especially for `04_mf_ratio_1.00` and `05_adaptive_off`.

> Inference-time NFE (1 vs 2 vs 4 segments) is **not** a training ablation — it
> re-samples the *same* trained checkpoint, so run it as a sweep over each
> checkpoint (cf. `flowcast_nfe_sweep.py`) rather than as a separate run here.

## Knobs deliberately NOT ablated here

These were considered and excluded because, in the current code, they cannot be
ablated by a Hydra override alone (a script for them would silently no-op):

- **`gap_embed_dim`** (Fourier width of the `(t − r)` interval embedding) is
  hardcoded to 32 in `MeanFlowPrecond` and not threaded through
  `get_preconditioned_architecture`, so `++model.gap_embed_dim=...` does nothing.
- **`valid_num_steps`** only sets the *validation* sampler NFE; it does not
  change training, so it belongs in an inference NFE sweep, not here.
- **`training.loss`** is read by the generic trainer but **not** by
  `train_meanflow.py` (which hardcodes `MeanFlowLoss`). `_common.sh` still passes
  `++training.loss=meanflow` purely for parity with the production launcher; it
  is inert and changing it would have no effect.
