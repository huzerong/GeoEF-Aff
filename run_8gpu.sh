#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GEOEF_AFF_PROJECT_ROOT="${GEOEF_AFF_PROJECT_ROOT:-${PACKAGE_ROOT}}"
export GEOEF_AFF_DATA_DIR="${GEOEF_AFF_DATA_DIR:-${PACKAGE_ROOT}/data}"
export PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-${GEOEF_AFF_DATA_DIR}/precomputed_samples}"
export RANDOM_SEED="${RANDOM_SEED:-42}"
export SPLIT_MODE="${SPLIT_MODE:-complex}"
export SPLIT_GROUP_COL="${SPLIT_GROUP_COL:-#Pdb}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PACKAGE_ROOT}/outputs/single_split_seed${RANDOM_SEED}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [[ ! -d "${PRECOMPUTED_DIR}" ]]; then
    echo "Precomputed sample directory not found: ${PRECOMPUTED_DIR}"
    echo "Set PRECOMPUTED_DIR or run preprocessing first."
    exit 1
fi

cd "${PACKAGE_ROOT}"
torchrun --standalone --nproc_per_node=8 main.py
