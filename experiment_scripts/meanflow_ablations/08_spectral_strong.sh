#!/bin/bash
# spectral_weight = 0.5  ->  5x the baseline log-PSD penalty on qpepre. Probes
# the other side of the spectral knob: a stronger term should tighten the
# small-scale power match, but may trade away pointwise RMSE/CSI if it pushes
# the model to hallucinate texture. Pairs with 07_spectral_off to bracket 0.1.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "08_spectral_strong" "++training.spectral_weight=0.5"
