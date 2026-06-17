#!/bin/bash
# Prune training checkpoints in a directory.
#
# Keeps weights whose step is a multiple of --keep-every (X) and, optionally,
# deletes all weights whose step is strictly greater than --max-step (Y).
# Everything else gets removed.
#
# Recognised filename formats (step is the second-to-last dot-separated field):
#   checkpoint.<run>.<step>.pt        e.g. checkpoint.0.140000.pt
#   <Name>.<run>.<step>.mdlus         e.g. EDMPrecond.0.70000.mdlus
#                                          FlowCastPrecond.0.140000.mdlus
#                                          StormCastUNet.0.8000.mdlus
#
# Default is DRY-RUN -- pass --apply to actually delete.
#
# Usage:
#   rm_weights.sh -d <dir> -k <keep_every> [-m <max_step>] [-p <glob>] [--apply]
#
# Examples:
#   # Show what would be removed (dry run): keep every 10k, drop nothing past Y
#   rm_weights.sh -d runs/.../checkpoints_flowcast -k 10000
#
#   # Keep every 25k AND drop anything past step 200000, then actually delete
#   rm_weights.sh -d runs/.../checkpoints_flowcast -k 25000 -m 200000 --apply
#
#   # Restrict to .mdlus files only
#   rm_weights.sh -d runs/.../checkpoints_diffusion -k 5000 -p '*.mdlus' --apply

set -euo pipefail

usage() {
    sed -n '2,30p' "$0"
    exit "${1:-1}"
}

dir=""
keep_every=""
max_step=""
pattern='*.pt *.mdlus'
apply=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir)        dir="$2"; shift 2 ;;
        -k|--keep-every) keep_every="$2"; shift 2 ;;
        -m|--max-step)   max_step="$2"; shift 2 ;;
        -p|--pattern)    pattern="$2"; shift 2 ;;
        --apply)         apply=1; shift ;;
        -h|--help)       usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
done

[[ -n "${dir}" ]]        || { echo "ERROR: -d/--dir is required" >&2; exit 1; }
[[ -n "${keep_every}" ]] || { echo "ERROR: -k/--keep-every is required" >&2; exit 1; }
[[ -d "${dir}" ]]        || { echo "ERROR: directory not found: ${dir}" >&2; exit 1; }
[[ "${keep_every}" =~ ^[0-9]+$ && "${keep_every}" -gt 0 ]] \
    || { echo "ERROR: --keep-every must be a positive integer" >&2; exit 1; }
if [[ -n "${max_step}" ]]; then
    [[ "${max_step}" =~ ^[0-9]+$ ]] \
        || { echo "ERROR: --max-step must be a non-negative integer" >&2; exit 1; }
fi

echo "Directory   : ${dir}"
echo "Keep every  : ${keep_every} steps"
echo "Max step    : ${max_step:-<none>}"
echo "Pattern     : ${pattern}"
echo "Mode        : $([[ ${apply} -eq 1 ]] && echo APPLY || echo 'DRY-RUN (pass --apply to delete)')"
echo

# Gather candidate files matching any of the space-separated globs.
# Split the pattern list with set -- so we DON'T accidentally apply pathname
# expansion to the literal glob tokens themselves (which, with nullglob on,
# would silently erase them when run from a directory that has no .pt/.mdlus).
read -r -a glob_list <<< "${pattern}"
files=()
shopt -s nullglob
for glob in "${glob_list[@]}"; do
    for f in "${dir}"/${glob}; do
        files+=("${f}")
    done
done
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No files matched in ${dir}"
    exit 0
fi

kept=()
removed=()
skipped=()

for f in "${files[@]}"; do
    base="$(basename "${f}")"
    # Step is the second-to-last dot-separated field: NAME.RUN.STEP.EXT
    step="$(echo "${base}" | awk -F. '{ print $(NF-1) }')"
    if ! [[ "${step}" =~ ^[0-9]+$ ]]; then
        skipped+=("${base} (could not parse step)")
        continue
    fi

    reason_keep=""
    reason_drop=""

    if [[ -n "${max_step}" && "${step}" -gt "${max_step}" ]]; then
        reason_drop="step ${step} > max ${max_step}"
    elif (( step % keep_every == 0 )); then
        reason_keep="step ${step} is a multiple of ${keep_every}"
    else
        reason_drop="step ${step} not a multiple of ${keep_every}"
    fi

    if [[ -n "${reason_keep}" ]]; then
        kept+=("${base}  (${reason_keep})")
    else
        removed+=("${base}  (${reason_drop})")
    fi
done

print_list() {
    local title="$1"; shift
    echo "=== ${title} (${#@}) ==="
    if [[ $# -eq 0 ]]; then
        echo "  <none>"
    else
        printf '  %s\n' "$@"
    fi
    echo
}

print_list "KEEP"    "${kept[@]:-}"
print_list "REMOVE"  "${removed[@]:-}"
[[ ${#skipped[@]} -gt 0 ]] && print_list "SKIPPED" "${skipped[@]}"

if [[ ${#removed[@]} -eq 0 ]]; then
    echo "Nothing to remove."
    exit 0
fi

if [[ ${apply} -eq 1 ]]; then
    echo "Deleting ${#removed[@]} file(s)..."
    for entry in "${removed[@]}"; do
        base="${entry%%  (*}"
        rm -f -- "${dir}/${base}"
    done
    echo "Done."
else
    echo "Dry-run only. Re-run with --apply to actually delete the ${#removed[@]} file(s) above."
fi
