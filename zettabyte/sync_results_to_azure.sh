#!/usr/bin/env bash
# Periodically mirror a local training-output directory on the zettabyte worker
# up to Azure Blob, so the same files can be pulled by the companion
# ``sync_results_from_azure.sh`` on a workstation.
#
# Run this on the zettabyte worker (where azcopy is preinstalled). Each tick
# uses ``azcopy sync``, which only transfers files whose size or mtime changed
# since the last run -- safe to run continuously.
#
# Usage:
#   ./sync_results_to_azure.sh [SRC] [REMOTE_REL]
#   SRC          local directory on zettabyte to mirror     (default: see below)
#   REMOTE_REL   destination prefix under the container     (default: see below)
#
# Environment:
#   INTERVAL     poll period in seconds                     (default: 300)
#   SYNC_FLAGS   extra flags to pass to azcopy sync         (default: empty)
#                e.g. SYNC_FLAGS="--exclude-pattern *.pt;*.mdlus" to skip
#                checkpoints if you only want logs/plots.
#
# Stop with Ctrl-C, or with ``kill <pid>`` if running under nohup.
#
# Run in foreground:
#   ./sync_results_to_azure.sh
# Run detached, with logs:
#   nohup ./sync_results_to_azure.sh > sync_to_azure.log 2>&1 &

set -uo pipefail

# Same SAS as azcopy_transfer.sh -- keep them in lockstep.
SAS_BASE="https://zbstore2026.blob.core.windows.net/g-019c8ca2-605d-7bb5-b98b-1c53fbdf2b7f"
SAS_QUERY="se=2026-09-14T06%3A47%3A36Z&sp=rwdl&sv=2026-02-06&sr=c&sig=oK8hMT%2BAJiBIzbozGc%2BgVk3N2e861Nm%2B9MvMyoi6UMg%3D"

# Defaults match the regression run wired in zettabyte_scripts/train_regression.sh.
DEFAULT_SRC="/data/exp_3_train_2_5_yrs_val_1yr_tp1/bridge_zettabyte_v1_cleaned_4_27_2026"
DEFAULT_REMOTE="desmond/runs/bridge_zettabyte_v1_cleaned_4_27_2026"

SRC="${1:-$DEFAULT_SRC}"
REMOTE_REL="${2:-$DEFAULT_REMOTE}"
INTERVAL="${INTERVAL:-300}"
SYNC_FLAGS="${SYNC_FLAGS:-}"

REMOTE_REL="${REMOTE_REL#/}"
REMOTE_REL="${REMOTE_REL%/}"     # strip any trailing slash so we control it
# Trailing '/' tells azcopy to treat both ends as directories (otherwise it
# tries to interpret the URL as a single blob and refuses to sync).
DEST_URL="${SAS_BASE}/${REMOTE_REL}/?${SAS_QUERY}"
SRC="${SRC%/}"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: source directory does not exist: $SRC" >&2
    echo "        (waiting -- it will be created when training starts)" >&2
fi

echo "[sync_to_azure] config:"
echo "    SRC          = $SRC"
echo "    REMOTE       = $REMOTE_REL"
echo "    INTERVAL     = ${INTERVAL}s"
echo "    SYNC_FLAGS   = ${SYNC_FLAGS:-<none>}"
echo "[sync_to_azure] starting poll loop -- Ctrl-C to stop"

trap 'echo "[sync_to_azure] stopping"; exit 0' INT TERM

while true; do
    ts="$(date '+%F %T')"
    if [[ -d "$SRC" ]]; then
        echo "[$ts] sync $SRC -> ${REMOTE_REL}/"
        # --recursive: walk the tree
        # --delete-destination=false: never remove blobs that vanished locally
        #   (azure stays the canonical archive even if the worker is rebuilt).
        # --put-md5: write content-MD5 so future syncs skip identical files
        #   without rechecking byte-by-byte.
        azcopy sync "${SRC}/" "$DEST_URL" \
            --recursive \
            --delete-destination=false \
            --put-md5 \
            $SYNC_FLAGS 2>&1 | tail -8
    else
        echo "[$ts] $SRC missing yet -- skipping this tick"
    fi
    sleep "$INTERVAL"
done
