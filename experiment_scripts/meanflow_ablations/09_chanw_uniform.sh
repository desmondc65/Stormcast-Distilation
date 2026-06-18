#!/bin/bash
# channel_weights = [1,1,1,1]  ->  removes the 2x boost on qpepre (last channel,
# order u10,v10,t2m,qpepre). Tests whether up-weighting the precip channel in
# the pointwise loss actually helps precip metrics, or whether the log1p
# transform already balances the channels enough that the boost is redundant.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "09_chanw_uniform" "++training.channel_weights=[1.0,1.0,1.0,1.0]"
