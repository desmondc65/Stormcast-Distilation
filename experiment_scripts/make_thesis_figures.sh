#!/bin/bash
# One-command regeneration of the thesis result artifacts.
#
# CPU-only (this script):
#   * make_thesis_figures.py  -> figures/results/{scoreboard_3way,rmse_per_channel,
#       crps_per_channel,csi_per_threshold,fss_p16,rollout_rmse,nfe_pareto}.png
#       (nfe_pareto only if the NFE sweep CSV exists)
#   * export_results_md.py     -> experiment_scripts/results.md
#
# GPU steps (run separately, need a free GPU):
#   * make_thesis_qualitative.sh   -> figures/results/qual_*.png  (viridis field panels)
#   * flowcast_nfe_sweep.py        -> results/flowcast_nfe_sweep/nfe_sweep.csv
#       (then re-run this script to pick up nfe_pareto.png)
#
# All colours follow thesis_style.py (viridis fields + viridis-sampled
# categorical palette), matching plot_weight_comparison_grid.py.

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
HERE="${REPO_ROOT}/experiment_scripts"

if [ -x "${REPO_ROOT}/stormcast_env/bin/python" ]; then
    PY="${REPO_ROOT}/stormcast_env/bin/python"
else
    PY="$(command -v python)"
fi

echo "[thesis-figs] python = ${PY}"
cd "${HERE}"
"${PY}" make_thesis_figures.py "$@"
"${PY}" export_results_md.py
echo "[thesis-figs] done. GPU panels: bash make_thesis_qualitative.sh"
echo "[thesis-figs]       NFE sweep: see run_flowcast_nfe_sweep.sh, then re-run this."
