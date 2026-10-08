#!/usr/bin/env bash
# Pre-train the latent world model on one node.
#
#   [NUM_GPUS=8] bash scripts/train_world_model.sh world_model/configs/world_model.yaml
#
# The batch size in the config is per GPU (8 GPUs in the paper). Checkpoints go
# to work_dirs/<config name>/; rerunning the command resumes from latest.pt.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${1:?usage: $0 <config.yaml>}"
NUM_GPUS="${NUM_GPUS:-8}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m world_model.train --config "${CONFIG}"
