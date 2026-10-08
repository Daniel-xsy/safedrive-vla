#!/usr/bin/env bash
# Train SafeDriveVLA on one node.
#
#   [NUM_GPUS=8] bash scripts/train_vla.sh <experiment> [hydra overrides...]
#   bash scripts/train_vla.sh safedrive_vla
#
# Experiments live in safedrive_vla/configs/experiment/. Outputs go to
# work_dirs/safedrive_vla/<experiment>/<timestamp>/. With RESUME=1 training
# continues from the newest complete epoch checkpoint of that experiment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

EXPERIMENT="${1:?usage: $0 <experiment> [hydra overrides...]}"
shift
NUM_GPUS="${NUM_GPUS:-8}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=("experiment=${EXPERIMENT}" "gpus=${NUM_GPUS}")
if [[ "${RESUME:-0}" == "1" ]]; then
  # DeepSpeed writes the `latest` tag last, so it marks a complete checkpoint.
  CKPT="$(for c in work_dirs/safedrive_vla/"${EXPERIMENT}"/*/checkpoints/epoch=*.ckpt; do
            if [[ -f "${c}/latest" ]]; then echo "$(basename "${c}") ${c}"; fi
          done | sort | tail -1 | cut -d' ' -f2-)"
  if [[ -n "${CKPT}" ]]; then
    echo "Resuming from ${CKPT}"
    ARGS+=("resume_path='${CKPT}'")
  fi
fi

python -m safedrive_vla.train "${ARGS[@]}" "$@"
