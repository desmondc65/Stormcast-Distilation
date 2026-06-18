#!/bin/bash
# ema_decay = 0.9999  ->  10x slower EMA than the 0.999 baseline (longer
# averaging window). MeanFlow inference uses the EMA shadow (ema_state.pt), so
# this knob directly moves the deployed weights. At a fixed 2M-sample budget a
# slower EMA averages over a larger fraction of training -> tests whether more
# averaging smooths the few-step samples or just lags the optimum.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "11_ema_0p9999" "++training.ema_decay=0.9999"
