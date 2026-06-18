#!/bin/bash
# mf_ratio = 0.50  ->  half the batch on the average-velocity identity, half on
# I-CFM. Upper-middle of the mf_ratio sweep {0.0, 0.25(base), 0.50, 0.75, 1.0};
# tests whether putting more of the batch on the bootstrapped target sharpens
# few-step samples or destabilizes training.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "02_mf_ratio_0p50" "++training.mf_ratio=0.5"
