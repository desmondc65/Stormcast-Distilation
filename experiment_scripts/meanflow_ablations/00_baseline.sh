#!/bin/bash
# Baseline: production MeanFlow recipe at the 2.0M-sample ablation budget.
# Every other script in this suite changes exactly one knob relative to this,
# so this run is the reference all deltas are read against.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "00_baseline"
