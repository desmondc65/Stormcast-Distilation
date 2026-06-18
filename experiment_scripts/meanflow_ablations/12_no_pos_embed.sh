#!/bin/bash
# spatial_pos_embed = False  ->  drops the additive learned spatial positional
# embedding from the SongUNet. The Taiwan domain is fixed (same lat/lon grid
# every sample), so a per-pixel embedding can memorize orography/coastline
# structure. This tests how much of the baseline skill comes from that spatial
# prior vs the conditioning channels (which already include the invariants).
source "$(dirname "$0")/_common.sh"
run_meanflow_ablation "12_no_pos_embed" "++model.spatial_pos_embed=False"
