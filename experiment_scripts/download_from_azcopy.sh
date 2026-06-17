#!/usr/bin/env bash
# =============================================================================
# download_from_azcopy.sh
#
# Fetch every model checkpoint the experiment harnesses in this directory
# (`experiment_scripts/`) load by default, pulling them from the Zettabyte
# zPLATFORM SAS blob container with AzCopy and mirroring them into the local
# repo tree so the `run_*.sh` launchers find them at their expected paths.
#
# The legacy old-stormcast checkpoints (exp_3_*) usually already live on disk;
# the missing pieces are the cleaned-4_27_2026 `runs/` checkpoints, which are
# what the cleaned EDM / FlowCast / MeanFlow legs need. This script downloads
# whatever is absent and skips whatever is already present.
#
# Which checkpoints, and which experiment needs them:
#   core     run_main_experiment.sh, run_single_time_exp.sh, run_inference_steps.sh,
#            run_qpw_ablation.sh, run_plot_rollout_12h.sh, make_thesis_qualitative.sh
#            -> regression 8000, EDM 31000, FlowCast 20000, MeanFlow 20000 (log1p, cleaned)
#   nfe      run_flowcast_nfe_sweep.sh
#            -> FlowCast 140000 (long-trained, for the NFE Pareto sweep)
#   nolog1p  run_log1p_ablation.sh  (the "Leg N" raw-mm/h ablation)
#            -> regression 10000, EDM 20000, FlowCast 20000 (NO_log1p, cleaned)
#   legacy   run_main_experiment.sh / run_plot_rollout_12h.sh "Leg A"
#            -> exp_3 regression 7500 + EDM 70000 (usually already on disk)
#
# -----------------------------------------------------------------------------
# Usage:
#   ./download_from_azcopy.sh [--dry-run] [--force] [GROUP ...]
#
#   GROUP    one or more of: core nfe nolog1p legacy all   (default: all)
#   --dry-run  list what would be fetched/skipped, transfer nothing
#   --force    re-download even if the local file already exists
#
# Examples:
#   ./download_from_azcopy.sh                 # everything that's missing
#   ./download_from_azcopy.sh core            # just the main-experiment 4
#   ./download_from_azcopy.sh --dry-run all   # show the full plan
#   ./download_from_azcopy.sh --force core    # re-pull the core 4
#
# -----------------------------------------------------------------------------
# Credentials (same convention as zettabyte/azcopy_transfer.sh -- secrets are
# read from the environment, never hard-coded into this tracked file):
#
#   export SAS_URL="https://<acct>.blob.core.windows.net/<container>?<sastoken>"
#     ...or split form...
#   export SAS_BASE="https://<acct>.blob.core.windows.net/<container>"
#   export SAS_QUERY="se=...&sp=...&sv=...&sr=c&sig=..."
#
# The exact line for this project is in zettabyte/zettabyte.md ("透過 AZCOPY").
# If neither is exported, the script sources the first of these it finds:
#   $AZCOPY_SAS_ENV, ~/.azcopy_sas.env, <repo>/zettabyte/.azcopy_sas.env
# (Keep any such file OUTSIDE git -- ~/.azcopy_sas.env is safest.)
#
# -----------------------------------------------------------------------------
# Env overrides:
#   DEST_ROOT    where the repo tree is mirrored (default: repo root, so files
#                land at $DEST_ROOT/runs/... exactly where the launchers look)
#   LOG_FILE     progress log (default: <script dir>/download_from_azcopy.log)
#   CAP_MBPS     throttle AzCopy bandwidth, e.g. CAP_MBPS=200
#   AZCOPY_BIN   azcopy binary name/path (default: azcopy)
#   USE_DOCKER=1 force the azcopy-local Docker image instead of a native azcopy
#
# Re-runnable: already-present files are skipped, so just run it again after a
# dropped connection. AzCopy also resumes partial single-file jobs internally.
# =============================================================================

set -uo pipefail   # NOT -e: one failed file must not abort the whole batch.

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DEST_ROOT="${DEST_ROOT:-${REPO_ROOT}}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/download_from_azcopy.log}"
# AzCopy's own verbose job/plan logs -- kept off the repo tree to avoid clutter.
AZCOPY_DIR="${AZCOPY_DIR:-${TMPDIR:-/tmp}/azcopy_stormcast_dl}"
AZCOPY_BIN="${AZCOPY_BIN:-azcopy}"

# --- Argument parsing --------------------------------------------------------
DRY_RUN=0
FORCE=0
GROUPS_REQUESTED=()
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        --force|-f)   FORCE=1 ;;
        -h|--help)    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        core|nfe|nolog1p|legacy|all) GROUPS_REQUESTED+=("$arg") ;;
        *) echo "ERROR: unknown argument '$arg' (try --help)" >&2; exit 2 ;;
    esac
done
[[ ${#GROUPS_REQUESTED[@]} -eq 0 ]] && GROUPS_REQUESTED=("all")

# --- Logging: mirror all output to console AND the log file ------------------
mkdir -p "$(dirname "$LOG_FILE")" "$AZCOPY_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

ts()   { date '+%Y-%m-%d %H:%M:%S'; }
log()  { printf '[%s] [dl] %s\n' "$(ts)" "$*"; }
warn() { printf '[%s] [dl][WARN] %s\n' "$(ts)" "$*"; }
banner() {
    printf '\n================================================================\n'
    printf '[%s] %s\n' "$(ts)" "$*"
    printf '================================================================\n'
}

banner "AzCopy checkpoint download"
log "log file   : ${LOG_FILE}"
log "repo root  : ${REPO_ROOT}"
log "dest root  : ${DEST_ROOT}    (files land at \$DEST_ROOT/runs/...)"
log "groups     : ${GROUPS_REQUESTED[*]}"
log "dry-run    : ${DRY_RUN}    force : ${FORCE}"
log "host       : $(hostname)  user=$(whoami)  pid=$$"

# --- Resolve SAS credentials -------------------------------------------------
# Source a gitignored env file only if SAS is not already in the environment.
if [[ -z "${SAS_URL:-}" && ( -z "${SAS_BASE:-}" || -z "${SAS_QUERY:-}" ) ]]; then
    for envf in "${AZCOPY_SAS_ENV:-}" "${HOME}/.azcopy_sas.env" \
                "${REPO_ROOT}/zettabyte/.azcopy_sas.env" "${SCRIPT_DIR}/.azcopy_sas.env"; do
        [[ -n "$envf" && -f "$envf" ]] || continue
        log "sourcing SAS credentials from ${envf}"
        set -a; # shellcheck disable=SC1090
        source "$envf"; set +a
        break
    done
fi
# Split a single SAS_URL into base + query if that's what was provided.
if [[ -n "${SAS_URL:-}" ]]; then
    SAS_BASE="${SAS_BASE:-${SAS_URL%%\?*}}"
    SAS_QUERY="${SAS_QUERY:-${SAS_URL#*\?}}"
fi
SAS_BASE="${SAS_BASE:-}"; SAS_QUERY="${SAS_QUERY:-}"
SAS_BASE="${SAS_BASE%/}"          # drop trailing slash
SAS_QUERY="${SAS_QUERY#\?}"       # drop a stray leading '?'
if [[ -z "$SAS_BASE" || -z "$SAS_QUERY" ]]; then
    warn "no SAS credentials found."
    warn "  export SAS_URL=\"https://<acct>.blob.core.windows.net/<container>?<token>\""
    warn "  (the exact line for this project is in zettabyte/zettabyte.md),"
    warn "  or place it in ~/.azcopy_sas.env -- then re-run."
    exit 1
fi
log "SAS base   : ${SAS_BASE}"

# Best-effort SAS-expiry check: parse se=<ISO8601> from the query and warn if
# it's already past (the #1 cause of silent AzCopy auth failures).
_se="$(printf '%s' "$SAS_QUERY" | tr '&' '\n' | sed -n 's/^se=//p' | sed 's/%3[Aa]/:/g')"
if [[ -n "$_se" ]]; then
    if _se_epoch="$(date -d "$_se" +%s 2>/dev/null)"; then
        if [[ "$_se_epoch" -lt "$(date +%s)" ]]; then
            warn "SAS token EXPIRED at ${_se} -- downloads will fail; refresh SAS_URL."
        else
            log "SAS expires : ${_se}"
        fi
    fi
fi

# --- Resolve the AzCopy runner (native binary preferred, Docker fallback) ----
AZCOPY_MODE=""
if [[ "${USE_DOCKER:-0}" != "1" ]] && command -v "$AZCOPY_BIN" >/dev/null 2>&1; then
    AZCOPY_MODE="native"
    log "azcopy     : $("$AZCOPY_BIN" --version 2>&1 | head -1) (native)"
elif docker image inspect azcopy-local >/dev/null 2>&1; then
    AZCOPY_MODE="docker"
    log "azcopy     : azcopy-local Docker image"
else
    warn "no usable azcopy: install the native binary (see zettabyte/zettabyte.md)"
    warn "  or build the azcopy-local Docker image, then re-run."
    exit 1
fi
export AZCOPY_LOG_LOCATION="$AZCOPY_DIR" AZCOPY_JOB_PLAN_LOCATION="$AZCOPY_DIR"

# azcopy_copy <src_url> <dst_file> -- copy one blob to one local path.
azcopy_copy() {
    local src="$1" dst="$2"
    local flags=(--overwrite=true --log-level INFO)
    [[ -n "${CAP_MBPS:-}" ]] && flags+=(--cap-mbps "${CAP_MBPS}")
    if [[ "$AZCOPY_MODE" == "native" ]]; then
        "$AZCOPY_BIN" copy "$src" "$dst" "${flags[@]}"
    else
        # Mount DEST_ROOT at the same absolute path so $dst resolves identically
        # inside the container. HOME=/tmp gives azcopy a writable state dir;
        # --user keeps the downloaded file owned by the host user.
        docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
            -v "$DEST_ROOT":"$DEST_ROOT" \
            azcopy-local \
            azcopy copy "$src" "$dst" "${flags[@]}"
    fi
}

# --- Manifest: group | remote-relative-path | local-relative-path ------------
# remote paths are relative to the container root (they begin with desmond/).
# local paths are relative to DEST_ROOT. NB the legacy pair maps a different
# remote prefix (desmond/exp_3_*) onto the local exp_3_train_2_5_yrs_* tree.
RUNS_REMOTE="desmond/runs"
MANIFEST=(
  # --- core: cleaned 192x96 + log1p, ~2M-sample budget (the headline legs) ---
  "core|${RUNS_REMOTE}/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus|runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus"
  "core|${RUNS_REMOTE}/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.31000.mdlus|runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.31000.mdlus"
  "core|${RUNS_REMOTE}/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus|runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus"
  "core|${RUNS_REMOTE}/meanflow_zettabyte_v1_cleaned_4_27_2026/meanflow_zettabyte_cleaned_4_27_2026/run_0/checkpoints_meanflow/MeanFlowPrecond.0.20000.mdlus|runs/meanflow_zettabyte_v1_cleaned_4_27_2026/meanflow_zettabyte_cleaned_4_27_2026/run_0/checkpoints_meanflow/MeanFlowPrecond.0.20000.mdlus"

  # --- nfe: long-trained FlowCast for the NFE Pareto sweep ---
  "nfe|${RUNS_REMOTE}/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.140000.mdlus|runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.140000.mdlus"

  # --- nolog1p: raw-mm/h ablation (Leg N of run_log1p_ablation.sh) ---
  "nolog1p|${RUNS_REMOTE}/regression_zettabyte_v1_cleaned_4_27_2026_NO_log1p/regression_cleaned_NO_log1p/run_0/checkpoints_regression/StormCastUNet.0.10000.mdlus|runs/regression_zettabyte_v1_cleaned_4_27_2026_NO_log1p/regression_cleaned_NO_log1p/run_0/checkpoints_regression/StormCastUNet.0.10000.mdlus"
  "nolog1p|${RUNS_REMOTE}/diffusion_zettabyte_v1_cleaned_4_27_2026_NO_log1p/edm_cleaned_NO_log1p/run_0/checkpoints_diffusion/EDMPrecond.0.20000.mdlus|runs/diffusion_zettabyte_v1_cleaned_4_27_2026_NO_log1p/edm_cleaned_NO_log1p/run_0/checkpoints_diffusion/EDMPrecond.0.20000.mdlus"
  "nolog1p|${RUNS_REMOTE}/flowcast_zettabyte_v1_cleaned_4_27_2026_NO_log1p/flowcast_cleaned_NO_log1p/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus|runs/flowcast_zettabyte_v1_cleaned_4_27_2026_NO_log1p/flowcast_cleaned_NO_log1p/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus"

  # --- legacy: old 224x128 raw-qpepre baseline (usually already on disk) ---
  "legacy|desmond/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.7500.mdlus|exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.7500.mdlus"
  "legacy|desmond/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus|exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus"
)

# group requested?  "all" matches everything.
want_group() {
    local g="$1" r
    for r in "${GROUPS_REQUESTED[@]}"; do
        [[ "$r" == "all" || "$r" == "$g" ]] && return 0
    done
    return 1
}

# --- Download loop -----------------------------------------------------------
TOTAL=0 DOWNLOADED=0 SKIPPED=0 FAILED=0
DL_BYTES=0
FAILED_LIST=()

banner "Plan"
for entry in "${MANIFEST[@]}"; do
    IFS='|' read -r grp remote_rel local_rel <<< "$entry"
    want_group "$grp" || continue
    dst="${DEST_ROOT}/${local_rel}"
    if [[ -s "$dst" && "$FORCE" != "1" ]]; then
        log "  present  [${grp}] ${local_rel}"
    else
        log "  fetch    [${grp}] ${local_rel}"
    fi
done

if [[ "$DRY_RUN" == "1" ]]; then
    banner "Dry run -- nothing transferred"
    exit 0
fi

banner "Downloading"
for entry in "${MANIFEST[@]}"; do
    IFS='|' read -r grp remote_rel local_rel <<< "$entry"
    want_group "$grp" || continue
    TOTAL=$((TOTAL + 1))
    dst="${DEST_ROOT}/${local_rel}"
    name="${local_rel##*/}"

    if [[ -s "$dst" && "$FORCE" != "1" ]]; then
        log "SKIP  [${grp}] ${name} -- already present ($(du -h "$dst" | cut -f1))"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    mkdir -p "$(dirname "$dst")"
    src="${SAS_BASE}/${remote_rel}?${SAS_QUERY}"
    log "GET   [${grp}] ${name}"
    log "      <- ${remote_rel}"
    log "      -> ${dst}"
    t0=$(date +%s)
    if azcopy_copy "$src" "$dst"; then
        t1=$(date +%s)
        if [[ -s "$dst" ]]; then
            bytes=$(stat -c %s "$dst" 2>/dev/null || echo 0)
            DL_BYTES=$((DL_BYTES + bytes))
            log "OK    [${grp}] ${name} ($(du -h "$dst" | cut -f1)) in $((t1 - t0))s"
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            warn "FAIL  [${grp}] ${name} -- azcopy exited 0 but no file at ${dst}"
            FAILED=$((FAILED + 1)); FAILED_LIST+=("${local_rel}")
        fi
    else
        rc=$?
        warn "FAIL  [${grp}] ${name} -- azcopy exit ${rc} (see ${AZCOPY_DIR} for details)"
        FAILED=$((FAILED + 1)); FAILED_LIST+=("${local_rel}")
    fi
done

# --- Summary -----------------------------------------------------------------
banner "Summary"
log "considered : ${TOTAL}"
log "downloaded : ${DOWNLOADED}  ($(numfmt --to=iec ${DL_BYTES} 2>/dev/null || echo "${DL_BYTES} B"))"
log "skipped    : ${SKIPPED}  (already present)"
log "failed     : ${FAILED}"
if [[ "$FAILED" -gt 0 ]]; then
    for f in "${FAILED_LIST[@]}"; do warn "  missing: ${f}"; done
    log "re-run the script to retry the failed files."
    exit 1
fi
log "all requested checkpoints are in place under ${DEST_ROOT}"
log "next: bash ${REPO_ROOT}/experiment_scripts/run_main_experiment.sh"
