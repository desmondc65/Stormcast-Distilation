#!/bin/bash
# channel_weights = [1,1,1,5]  ->  pushes the qpepre weight from 2x to 5x (order
# u10,v10,t2m,qpepre). Together with 09_chanw_uniform this brackets the precip
# weighting at {1x, 2x(base), 5x} so the trade-off between precip skill and the
# three smooth channels (t2m/u10/v10 RMSE) can be quantified.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "10_chanw_qpepre_strong" "++training.channel_weights=[1.0,1.0,1.0,5.0]"
