#!/usr/bin/env bash
# Simple azcopy wrapper for uploads/downloads to the zPLATFORM SAS container.
#
# Usage:
#   ./azcopy_transfer.sh up   <local_src>        <remote_dst_relative>
#   ./azcopy_transfer.sh down <remote_src_relative> <local_dst>
#   ./azcopy_transfer.sh list [--depth N] [remote_relative]
#
# Examples:
#   ./azcopy_transfer.sh up   /data/exp_3/progressive_zettabyte_v1  desmond/results/progressive
#   ./azcopy_transfer.sh down desmond/results/progressive/progressive_zettabyte_v1  ./downloads
#   ./azcopy_transfer.sh list desmond/results/progressive
#   ./azcopy_transfer.sh list --depth 1 desmond/results          # direct children only
#   ./azcopy_transfer.sh list --depth 2 desmond/results          # two levels deep
#   ./azcopy_transfer.sh list -d 1                               # top-level of container
#
# Notes:
#   - Folders are transferred recursively automatically.
#   - Append "/*" to a local source to upload folder contents without the parent dir.
#   - Remote paths are relative to the container root (no leading slash).
#   - --depth (or -d) filters listed blobs to at most N path segments below the prefix.
#     Depth 1 = direct children, 2 = one level of subdirectories, etc.

set -euo pipefail

SAS_BASE="https://zbstore2026.blob.core.windows.net/g-019c8ca2-605d-7bb5-b98b-1c53fbdf2b7f"
SAS_QUERY="se=2026-09-14T06%3A47%3A36Z&sp=rwdl&sv=2026-02-06&sr=c&sig=oK8hMT%2BAJiBIzbozGc%2BgVk3N2e861Nm%2B9MvMyoi6UMg%3D"

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
        depth=""
        remote_rel=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --depth|-d)
                    shift
                    [[ $# -gt 0 ]] || { echo "ERROR: --depth requires a numeric argument" >&2; exit 1; }
                    depth="$1"
                    ;;
                *)
                    remote_rel="$1"
                    ;;
            esac
            shift
        done

        if [[ -z "$depth" ]]; then
            azcopy list "$(remote_url "$remote_rel")"
        else
            # azcopy list outputs lines as: "<blob-path>; Content Length: ..."
            # Paths are already relative to the listed URL prefix, so no prefix stripping needed.
            azcopy list "$(remote_url "$remote_rel")" | awk -v depth="$depth" '
                {
                    sc = index($0, ";")
                    blob = (sc > 0) ? substr($0, 1, sc - 1) : $0
                    gsub(/[[:space:]]+$/, "", blob)
                    n_slashes = split(blob, parts, "/") - 1
                    if (n_slashes < depth) print $0
                }
            '
        fi
        ;;
    *)
        usage
        ;;
esac
