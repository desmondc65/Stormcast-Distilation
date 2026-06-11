#!/bin/bash
# Regenerate the qualitative field panels used in Chapter 4 of the thesis, in
# the viridis palette (matching plot_weight_comparison_grid.py / validation_plot).
#
# These are the truth | diffusion (mean) | flowcast (mean) field panels plus the
# (pred - truth) difference row, for a couple of 2022 validation sequences:
#
#   figures/results/qual_qpepre_seq00.png   precipitation, validation sequence A
#   figures/results/qual_qpepre_seq02.png   precipitation, validation sequence B
#   figures/results/qual_u10_seq00.png      10 m zonal wind, validation sequence A
#
# They come straight from compare_diffusion_vs_flowcast.py's plot_panels(), which
# now reads its field colormap from thesis_style (viridis). We run the cleaned
# main-experiment leg at the SAME sequence selection as the scoreboard
# (n_sequences=24, seed=0) but only the +1h step, so seq00/seq02 are the same
# validation cases shown in the headline table — just rendered in viridis.
#
# GPU run (~1-2 min on a free A6000). Override CUDA_VISIBLE_DEVICES / OUT_DIR /
# checkpoints via the environment.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

if [ -x "${REPO_ROOT}/stormcast_env/bin/python" ]; then
    PY="${REPO_ROOT}/stormcast_env/bin/python"
else
    PY="$(command -v python)"
fi

# 0 is usually training on this box; 1/2 are free.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiment_scripts/results/qual_thesis}"
FIG_OUT="${FIG_OUT:-${REPO_ROOT}/NTU-Thesis-LaTeX-Template/figures/results}"

# Cleaned main-experiment leg checkpoints (match run_main_experiment.sh / the
# headline scoreboard: cleaned EDM @ ~2M = step 31000, FlowCast @ ~2M = step 20000).
DATA="${DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
CLEANED_REG="${CLEANED_REG:-${REPO_ROOT}/runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus}"
CLEANED_EDM="${CLEANED_EDM:-${REPO_ROOT}/runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.31000.mdlus}"
CLEANED_FLOW="${CLEANED_FLOW:-${REPO_ROOT}/runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus}"

N_SEQUENCES="${N_SEQUENCES:-24}"   # match the scoreboard's sequence grid
ENSEMBLE="${ENSEMBLE:-10}"          # ensemble mean shown in the (mean) panels (10-member standard)
SEED="${SEED:-0}"

echo "[qual] python       = ${PY}"
echo "[qual] CUDA devices = ${CUDA_VISIBLE_DEVICES}"
echo "[qual] out_dir      = ${OUT_DIR}"
echo "[qual] fig_out      = ${FIG_OUT}"

mkdir -p "${OUT_DIR}" "${FIG_OUT}"

# Only the +1h step (panel-steps 0); n_panels_seq=3 renders seq00/01/02.
"${PY}" -u "${REPO_ROOT}/experiment_scripts/compare_diffusion_vs_flowcast.py" \
    --data-location "${DATA}" \
    --hr-size 192 96 --qpepre-log1p --kept-channels u10 v10 t2m qpepre \
    --regression-checkpoint "${CLEANED_REG}" \
    --diffusion-checkpoint "${CLEANED_EDM}" \
    --flowcast-checkpoint "${CLEANED_FLOW}" \
    --output-dir "${OUT_DIR}" \
    --n-sequences "${N_SEQUENCES}" --n-steps 1 --ensemble "${ENSEMBLE}" \
    --seed "${SEED}" \
    --n-panels-seq 3 --panel-steps 0 \
    2>&1 | tee "${OUT_DIR}/run.log"

# Copy the chosen panels to their thesis filenames.
cp -f "${OUT_DIR}/panels/qpepre/seq00_step00.png" "${FIG_OUT}/qual_qpepre_seq00.png"
cp -f "${OUT_DIR}/panels/qpepre/seq02_step00.png" "${FIG_OUT}/qual_qpepre_seq02.png"
cp -f "${OUT_DIR}/panels/u10/seq00_step00.png"    "${FIG_OUT}/qual_u10_seq00.png"

echo "[qual] copied 3 panels into ${FIG_OUT}:"
echo "       qual_qpepre_seq00.png  qual_qpepre_seq02.png  qual_u10_seq00.png"
echo "[qual] done"
