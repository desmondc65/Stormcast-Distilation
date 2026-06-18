#!/bin/bash
# mf_ratio = 0.75  ->  most of the batch on the average-velocity identity, only
# a quarter on the I-CFM anchor. Probes how far the ratio can be pushed before
# the shrinking flow-matching anchor lets the JVP target drift.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "03_mf_ratio_0p75" "++training.mf_ratio=0.75"
