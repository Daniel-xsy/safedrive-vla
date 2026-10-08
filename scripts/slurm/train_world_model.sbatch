#!/usr/bin/env bash
# Slurm wrapper of scripts/train_world_model.sh (one node). Activate the conda
# environment before submitting; add -p/-A/-t for your cluster, e.g.
#
#   sbatch -p <partition> scripts/slurm/train_world_model.sbatch world_model/configs/world_model.yaml
#
#SBATCH --job-name=safedrive-wm
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# The launcher starts one process per GPU itself (Lightning / torchrun);
# hide the Slurm task layout so it is not mistaken for an srun step.
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID
NUM_GPUS="$(nvidia-smi -L | grep -c '^GPU ')" bash scripts/train_world_model.sh "$@"
