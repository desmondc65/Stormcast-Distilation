#!/usr/bin/env bash
# Periodically pull training results from Azure Blob into a local directory on
# this workstation, so logs / loss CSVs / plots / checkpoints land here as
# they show up on the zettabyte worker (the upstream side runs
# ``sync_results_to_azure.sh``).
#
# Uses azcopy via the ``azcopy-local`` Docker image (this server cannot install
# azcopy natively). Each tick runs ``azcopy sync``, which only transfers files
# whose size or mtime changed since the last run.
#
# Usage:
#   ./sync_results_from_azure.sh [REMOTE_REL] [DST]
#   REMOTE_REL   source prefix under the container      (default: see below)
#   DST          local directory to mirror into          (default: see below)
#
# Environment:
#   INTERVAL     poll period in seconds                  (default: 300)
#   SYNC_FLAGS   extra flags to pass to azcopy sync      (default: empty)
#                e.g. SYNC_FLAGS="--exclude-pattern *.pt" to skip checkpoints.
#
# Stop with Ctrl-C, or with ``kill <pid>`` if running under nohup.
#
# Run in foreground:
#   ./sync_results_from_azure.sh
# Run detached, with logs:
#   nohup ./sync_results_from_azure.sh > sync_from_azure.log 2>&1 &

set -uo pipefail

# Same SAS as azcopy_transfer.sh and sync_results_to_azure.sh -- read from
# the environment so secrets never live in git. Export SAS_BASE and
# SAS_QUERY in your shell (e.g. source a gitignored ~/.azcopy_sas.env)
# before invoking this script.
: "${SAS_BASE:?set SAS_BASE in env (container URL, no query string)}"
: "${SAS_QUERY:?set SAS_QUERY in env (SAS token query string, no leading '?')}"


DEFAULT_REMOTE="desmond/runs/bridge_zettabyte_v1_cleaned_4_27_2026"
DEFAULT_DST="/home3/davidlcs/Econ-Rag/Local_LLM/test_meeting/Stormcast-Distilation/runs/bridge_zettabyte_v1_cleaned_4_27_2026"

REMOTE_REL="${1:-$DEFAULT_REMOTE}"
DST="${2:-$DEFAULT_DST}"
INTERVAL="${INTERVAL:-300}"
SYNC_FLAGS="${SYNC_FLAGS:-}"

# Loss-curve plotting (re-rendered after every successful sync).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_SCRIPT="${PLOT_SCRIPT:-$SCRIPT_DIR/plot_training_curves.py}"
PLOT_PYTHON="${PLOT_PYTHON:-/home3/davidlcs/.conda/envs/simple-rag/bin/python}"
PLOT_RECENT="${PLOT_RECENT:-2000}"      # window for the right-panel "recent" zoom
PLOT_DISABLE="${PLOT_DISABLE:-0}"        # set to 1 to skip plotting entirely

REMOTE_REL="${REMOTE_REL#/}"
REMOTE_REL="${REMOTE_REL%/}"     # strip any trailing slash so we control it
# Trailing '/' tells azcopy to treat both ends as directories (otherwise it
# tries to interpret the URL as a single blob and refuses to sync).
SRC_URL="${SAS_BASE}/${REMOTE_REL}/?${SAS_QUERY}"

mkdir -p "$DST"
DST="${DST%/}"                   # normalize -- we add the slash on the docker mount

# Sanity: confirm the azcopy-local image is built.
if ! docker image inspect azcopy-local >/dev/null 2>&1; then
    echo "ERROR: docker image 'azcopy-local' is missing." >&2
    echo "        Build it with the recipe in zettabyte/zettabyte.md (Docker section)." >&2
    exit 1
fi

echo "[sync_from_azure] config:"
echo "    REMOTE       = $REMOTE_REL"
echo "    DST          = $DST"
echo "    INTERVAL     = ${INTERVAL}s"
echo "    SYNC_FLAGS   = ${SYNC_FLAGS:-<none>}"
if [[ "$PLOT_DISABLE" == "1" ]]; then
    echo "    PLOT         = disabled"
else
    echo "    PLOT         = $PLOT_PYTHON $PLOT_SCRIPT  (recent=${PLOT_RECENT})"
fi
echo "[sync_from_azure] starting poll loop -- Ctrl-C to stop"

trap 'echo "[sync_from_azure] stopping"; exit 0' INT TERM

while true; do
    ts="$(date '+%F %T')"
    echo "[$ts] sync ${REMOTE_REL}/ -> $DST"
    # --recursive: walk the tree.
    # --delete-destination=false: never remove local files that vanished on
    #   Azure (a stale local copy is preferable to data loss).
    # --user keeps downloaded files owned by the host user (not root) so the
    # post-sync plotter can overwrite anything inside DST without sudo.
    # HOME=/tmp gives azcopy a writable place for its log/.azcopy dir; without
    # this it tries to mkdir /.azcopy and fails with "permission denied",
    # which silently aborts the sync (no bytes transferred, no Final Job
    # Status line in the output).
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -v "$DST":/workspace/dst \
        azcopy-local \
        azcopy sync "$SRC_URL" "/workspace/dst/" \
            --recursive \
            --delete-destination=false \
            $SYNC_FLAGS 2>&1 | tail -8

    # Re-render loss curves after each tick. Failure here must not break the
    # sync loop, so we swallow non-zero exits.
    if [[ "$PLOT_DISABLE" != "1" && -x "$PLOT_PYTHON" && -f "$PLOT_SCRIPT" ]]; then
        "$PLOT_PYTHON" "$PLOT_SCRIPT" "$DST" --recent "$PLOT_RECENT" || true
    fi

    sleep "$INTERVAL"
done
