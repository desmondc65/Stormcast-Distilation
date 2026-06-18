#!/bin/bash
# attn_resolutions = [24]  ->  adds self-attention at the 24-px U-Net level
# (resolutions are res = 192>>level in {192,96,48,24,12}; baseline uses [], no
# attention). Self-attention adds global context, which can help organize
# spatially-coherent convective precip structures that a purely-convolutional
# net handles only locally. Costs more memory/compute -- drop batch_size_per_gpu
# in _common.sh if a worker OOMs.
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "13_attn_24" "++model.attn_resolutions=[24]"
