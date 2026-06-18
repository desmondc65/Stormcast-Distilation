#!/bin/bash
# mf_ratio = 0.0  ->  pure instantaneous flow matching (I-CFM), NO average-velocity
# bootstrap. This is the key A/B: it isolates whether the MeanFlow identity
# (the JVP-bootstrapped r<t target) is what buys few-step quality, vs plain
# flow matching evaluated at 1-2 NFE. Expect this to be the weakest at 1 NFE if
# MeanFlow is doing its job.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "01_mf_ratio_0p00" "++training.mf_ratio=0.0"
