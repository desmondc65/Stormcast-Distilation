#!/usr/bin/env bash
# Upload the finished no-spectral MeanFlow run (train_meanflow_no_lspec.sh)
# from the zettabyte worker up to Azure Blob, so it can be pulled back to a
# workstation with the companion ../../zettabyte/sync_results_from_azure.sh.
#
# Run this ON the zettabyte worker (azcopy is preinstalled there) AFTER
# train_meanflow_no_lspec.sh has finished. It is a single idempotent pass:
# `azcopy sync` only transfers files whose size/mtime changed, so it is safe to
# re-run without re-uploading everything.
#
# What it uploads: the whole run tree written by train_meanflow_no_lspec.sh
#   <SRC>/<exp>/run_0/{checkpoints_meanflow,*.csv,images/,hydra/}
# i.e. the single MeanFlowPrecond.0.18000.mdlus (+ ema_state.pt) plus every
# loss/RMSE/PS1D CSV and validation heatmap. (Source default matches
# training_output_dir in train_meanflow_no_lspec.sh.)
#
# Credentials: export the same SAS_URL you already used to pull the dataset
# (see ../../zettabyte/zettabyte.md) -- the full container URL incl. query:
#   export SAS_URL="https://zbstore2026.blob.core.windows.net/g-019c8ca2-...?se=...&sig=..."
# Alternatively export the split SAS_BASE + SAS_QUERY pair that the
# ../../zettabyte/*.sh scripts use; either form works.
#
# Usage:
#   ./upload_meanflow_no_lspec.sh [SRC] [REMOTE_REL]
#   SRC          local dir on the worker to mirror   (default: training_output_dir below)
#   REMOTE_REL   destination prefix in the container (default: desmond/runs/<basename>)
#
# Environment:
#   SYNC_FLAGS   extra flags forwarded to azcopy sync (default: empty)
#                e.g. SYNC_FLAGS="--exclude-pattern *.mdlus;*.pt" to ship only
#                logs/CSVs/plots and skip the heavy checkpoints.

set -uo pipefail

# --- Resolve SAS credentials -------------------------------------------------
# Prefer the single SAS_URL (the worker already has this exported for the
# dataset pull); fall back to the SAS_BASE + SAS_QUERY split used by the
# zettabyte/*.sh scripts. Secrets stay in the env, never in git.
if [[ -n "${SAS_URL:-}" ]]; then
    SAS_BASE="${SAS_BASE:-${SAS_URL%%\?*}}"   # strip ?query -> container URL
    SAS_QUERY="${SAS_QUERY:-${SAS_URL#*\?}}"  # strip URL? -> token query string
fi
: "${SAS_BASE:?set SAS_URL (full container SAS) or SAS_BASE (container URL, no query)}"
: "${SAS_QUERY:?set SAS_URL (full container SAS) or SAS_QUERY (SAS token query, no leading '?')}"

# --- Defaults: match training_output_dir in train_meanflow_no_lspec.sh -------
DEFAULT_SRC="/data/exp_3_train_2_5_yrs_val_1yr_tp1/meanflow_no_lspec_zettabyte_v1_cleaned_4_27_2026"

SRC="${1:-$DEFAULT_SRC}"
SRC="${SRC%/}"
# Default remote keeps the same basename under desmond/runs/, matching the
# convention in zettabyte/sync_results_to_azure.sh + sync_results_from_azure.sh.
DEFAULT_REMOTE="desmond/runs/$(basename "$SRC")"
REMOTE_REL="${2:-$DEFAULT_REMOTE}"
SYNC_FLAGS="${SYNC_FLAGS:-}"

REMOTE_REL="${REMOTE_REL#/}"
REMOTE_REL="${REMOTE_REL%/}"     # strip any trailing slash so we control it
# Trailing '/' on both ends tells azcopy to treat them as directories (otherwise
# it tries to interpret the URL as a single blob and refuses to sync).
DEST_URL="${SAS_BASE}/${REMOTE_REL}/?${SAS_QUERY}"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: source directory does not exist: $SRC" >&2
    echo "        (run train_meanflow_no_lspec.sh first, or pass SRC explicitly)" >&2
    exit 1
fi

echo "[upload_no_lspec] config:"
echo "    SRC          = $SRC"
echo "    REMOTE       = $REMOTE_REL"
echo "    SYNC_FLAGS   = ${SYNC_FLAGS:-<none>}"
echo "[upload_no_lspec] uploading $SRC -> ${REMOTE_REL}/ ..."

# --recursive: walk the tree.
# --delete-destination=false: never remove blobs that vanished locally, so Azure
#   stays the canonical archive even if the worker is rebuilt.
# --put-md5: write content-MD5 so future syncs skip identical files without
#   rechecking byte-by-byte.
azcopy sync "${SRC}/" "$DEST_URL" \
    --recursive \
    --delete-destination=false \
    --put-md5 \
    $SYNC_FLAGS
rc=$?

if [[ $rc -ne 0 ]]; then
    echo "[upload_no_lspec] azcopy sync FAILED (exit $rc)" >&2
    exit $rc
fi

echo "[upload_no_lspec] done. Verify the remote tree with:"
echo "    azcopy list \"${SAS_BASE}/${REMOTE_REL}/?\$SAS_QUERY\""
echo "[upload_no_lspec] pull it back to a workstation with:"
echo "    ./sync_results_from_azure.sh ${REMOTE_REL} <local_dst>   # in ../../zettabyte/"
