#!/bin/bash
# adaptive_p = 0.0  ->  disables the adaptive loss weight w = 1/(mse+eps)^p and
# trains on a plain (channel-weighted) MSE. The adaptive weight is one of the
# headline MeanFlow tricks (down-weights high-error samples to stabilize the
# bootstrap). On heavy-tailed qpepre it could either help or wash out the rare
# heavy-rain pixels -- this run measures which.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "05_adaptive_off" "++training.adaptive_p=0.0"
