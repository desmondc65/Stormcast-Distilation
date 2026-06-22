#!/bin/bash
# Run the MeanFlow ablation suite sequentially. All 7 GPUs are used by each run,
# so runs go one at a time. Each run already saves a single checkpoint at ~2M
# samples; this wrapper just chains them and skips any whose final checkpoint
# already exists (so it is safe to re-run after an interruption).
#
# Usage:
#   ./run_all.sh             # core subset (recommended first pass)
#   ./run_all.sh all         # every ablation in this directory
#   ./run_all.sh core        # explicit core subset
#   ./run_all.sh extended    # only the bracketing / secondary runs
#   ./run_all.sh 01 05 13    # only the scripts whose names start with these
#
# Groups (see README.md for the rationale of each):
#   core     : 00 baseline, the mf_ratio sweep, adaptive off, spectral off,
#              channel-weight uniform -- the questions most central to the method.
#   extended : 06 adaptive_p0.5, 08 spectral_strong, 10 chanw_qpepre_strong,
#              11 ema_0.9999, 12 no_pos_embed, 13 attn -- the bracketing /
#              secondary runs.
#   all      : core + extended.

cd "$(dirname "$0")"
# _common.sh enables `set -euo pipefail` (right for a single training run); the
# orchestrator does its own error handling and intentionally tolerates non-zero
# exits from the grep/compgen probes below, so relax those options after sourcing.
source "./_common.sh"
set +e +o pipefail

CORE=(00_baseline 01_mf_ratio_0.00 02_mf_ratio_0.50 03_mf_ratio_0.75 \
      04_mf_ratio_1.00 05_adaptive_off 07_spectral_off 09_chanw_uniform)
EXTENDED=(06_adaptive_p0.5 08_spectral_strong 10_chanw_qpepre_strong \
          11_ema_0.9999 12_no_pos_embed 13_attn)

selection=()
case "${1:-core}" in
  all)      selection=("${CORE[@]}" "${EXTENDED[@]}") ;;
  core)     selection=("${CORE[@]}") ;;
  extended) selection=("${EXTENDED[@]}") ;;
  *)    # treat args as name prefixes to filter all scripts
        for arg in "$@"; do
          for f in [0-9][0-9]_*.sh; do
            [[ "$f" == "${arg}"* ]] && selection+=("${f%.sh}")
          done
        done ;;
esac

echo "Will run ${#selection[@]} ablation(s): ${selection[*]}"

for name in "${selection[@]}"; do
  script="${name}.sh"
  if [[ ! -f "$script" ]]; then
    echo "[skip] $script not found"; continue
  fi
  # experiment_name is the script's run_meanflow_ablation label; the final
  # checkpoint lands at <ABLATION_BASE>/<exp>/run_0/checkpoints_meanflow.
  exp="$(grep -oE 'run_meanflow_ablation "[^"]+"' "$script" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
  ckpt_dir="${ABLATION_BASE}/${exp}/run_${RUN_ID}/checkpoints_meanflow"
  if compgen -G "${ckpt_dir}/MeanFlowPrecond.*.mdlus" > /dev/null 2>&1; then
    echo "[done] ${exp} already has a checkpoint in ${ckpt_dir} -- skipping"
    continue
  fi
  echo ">>> launching ${script} (${exp})"
  bash "$script"
done

echo "All requested ablations finished."
