#!/bin/bash
# spectral_weight = 0.0  ->  removes the radial log-PSD regularizer on qpepre.
# Tests how much the spectral term contributes to precip realism: with it off,
# expect the qpepre power spectrum to roll off (over-smooth small scales) even
# if pointwise RMSE looks similar. Read against the ps1d_qpepre.csv / spectra.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "07_spectral_off" "++training.spectral_weight=0.0"
