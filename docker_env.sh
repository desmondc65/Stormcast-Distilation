#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-stormcast-ncdr:dev}"
CONTAINER_NAME="${CONTAINER_NAME:-stormcast-ncdr-dev}"
WORKDIR_IN_CONTAINER="${WORKDIR_IN_CONTAINER:-/workspace}"
GPU_FLAG="${GPU_FLAG:---gpus all}"

usage() {
  cat <<'EOF'
Usage:
  ./docker_env.sh build    # Build image
  ./docker_env.sh run      # Start interactive container (workspace mounted)
  ./docker_env.sh shell    # Enter a running container shell

Optional environment variables:
  IMAGE_NAME=stormcast-ncdr:dev
  CONTAINER_NAME=stormcast-ncdr-dev
  WORKDIR_IN_CONTAINER=/workspace
  GPU_FLAG='--gpus all'    # set GPU_FLAG='' for CPU-only run
EOF
}

build_image() {
  docker build -t "${IMAGE_NAME}" .
}

run_container() {
  docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    ${GPU_FLAG} \
    -v "${PWD}:${WORKDIR_IN_CONTAINER}" \
    -w "${WORKDIR_IN_CONTAINER}" \
    "${IMAGE_NAME}"
}

shell_container() {
  docker exec -it "${CONTAINER_NAME}" bash
}

# Default behavior: run container when no subcommand is provided.
cmd="${1:-run}"

case "${cmd}" in
  -h|--help|help)
    usage
    ;;
  build)
    build_image
    ;;
  run)
    run_container
    ;;
  shell)
    shell_container
    ;;
  *)
    usage
    exit 1
    ;;
esac