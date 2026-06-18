#!/bin/bash
# mf_ratio = 1.0  ->  the ENTIRE batch on the average-velocity identity, no
# I-CFM anchor at all. Geng et al. flag this as the unstable extreme (the loss
# can lose its grounding in the instantaneous velocity). Including it documents
# the failure mode and the cost of dropping the flow-matching anchor; watch for
# divergence / precip mode collapse in the validation curves.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "04_mf_ratio_1p00" "++training.mf_ratio=1.0"
