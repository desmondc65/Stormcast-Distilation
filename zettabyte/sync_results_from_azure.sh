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

# Same SAS as azcopy_transfer.sh and sync_results_to_azure.sh.
SAS_BASE="https://zbstore2026.blob.core.windows.net/g-019c8ca2-605d-7bb5-b98b-1c53fbdf2b7f"
SAS_QUERY="se=2026-09-14T06%3A47%3A36Z&sp=rwdl&sv=2026-02-06&sr=c&sig=oK8hMT%2BAJiBIzbozGc%2BgVk3N2e861Nm%2B9MvMyoi6UMg%3D"


DEFAULT_REMOTE="desmond/runs/"
DEFAULT_DST="/home/desmond/Documents/master_thesis/Stormcast-Distilation/runs/"


REMOTE_REL="${1:-$DEFAULT_REMOTE}"
DST="${2:-$DEFAULT_DST}"
INTERVAL="${INTERVAL:-300}"
SYNC_FLAGS="${SYNC_FLAGS:-}"

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
echo "[sync_from_azure] starting poll loop -- Ctrl-C to stop"

trap 'echo "[sync_from_azure] stopping"; exit 0' INT TERM

while true; do
    ts="$(date '+%F %T')"
    echo "[$ts] sync ${REMOTE_REL}/ -> $DST"
    # --recursive: walk the tree.
    # --delete-destination=false: never remove local files that vanished on
    #   Azure (a stale local copy is preferable to data loss).
    docker run --rm \
        -v "$DST":/workspace/dst \
        azcopy-local \
        azcopy sync "$SRC_URL" "/workspace/dst/" \
            --recursive \
            --delete-destination=false \
            $SYNC_FLAGS 2>&1 | tail -8
    sleep "$INTERVAL"
done
