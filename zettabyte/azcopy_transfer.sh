#!/usr/bin/env bash
# Simple azcopy wrapper for uploads/downloads to the zPLATFORM SAS container.
#
# Usage:
#   ./azcopy_transfer.sh up   <local_src>        <remote_dst_relative>
#   ./azcopy_transfer.sh down <remote_src_relative> <local_dst>
#   ./azcopy_transfer.sh list [remote_relative]
#
# Examples:
#   ./azcopy_transfer.sh up   /data/exp_3/progressive_zettabyte_v1  desmond/results/progressive
#   ./azcopy_transfer.sh down desmond/results/progressive/progressive_zettabyte_v1  ./downloads
#   ./azcopy_transfer.sh list desmond/results/progressive
#
# Notes:
#   - Folders are transferred recursively automatically.
#   - Append "/*" to a local source to upload folder contents without the parent dir.
#   - Remote paths are relative to the container root (no leading slash).

set -euo pipefail

# SAS credentials come from the environment so they never live in git.
# Export SAS_BASE and SAS_QUERY in your shell (e.g. from a gitignored
# ~/.azcopy_sas.env you ``source``) before invoking this script.
: "${SAS_BASE:?set SAS_BASE in env (container URL, no query string)}"
: "${SAS_QUERY:?set SAS_QUERY in env (SAS token query string, no leading '?')}"

remote_url() {
    local rel="${1:-}"
    rel="${rel#/}"
    if [[ -z "$rel" ]]; then
        echo "${SAS_BASE}?${SAS_QUERY}"
    else
        echo "${SAS_BASE}/${rel}?${SAS_QUERY}"
    fi
}

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

cmd="${1:-}"; shift || usage

case "$cmd" in
    up|upload)
        [[ $# -eq 2 ]] || usage
        src="$1"; dst_rel="$2"
        azcopy copy "$src" "$(remote_url "$dst_rel")" --recursive
        ;;
    down|download)
        [[ $# -eq 2 ]] || usage
        src_rel="$1"; dst="$2"
        mkdir -p "$dst"
        azcopy copy "$(remote_url "$src_rel")" "$dst" --recursive
        ;;
    list|ls)
        azcopy list "$(remote_url "${1:-}")"
        ;;
    *)
        usage
        ;;
esac
