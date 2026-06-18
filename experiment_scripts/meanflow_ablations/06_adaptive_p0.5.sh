#!/bin/bash
# adaptive_p = 0.5  ->  milder adaptive weighting between off (0.0) and the
# paper/base default (1.0). Fills in the adaptive_p sweep so the effect can be
# read as a trend rather than a single on/off contrast.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "06_adaptive_p0p5" "++training.adaptive_p=0.5"
