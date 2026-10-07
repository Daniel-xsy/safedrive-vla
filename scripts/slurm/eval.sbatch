#!/usr/bin/env bash
# Slurm wrapper of scripts/eval.sh (one node, one CARLA server per GPU).
# Activate the conda environment and export CARLA_ROOT before submitting, e.g.
#
#   sbatch -p <partition> scripts/slurm/eval.sbatch bench2drive <checkpoint> [eval.sh options]
#
# Resubmitting the same command resumes the evaluation.
#
#SBATCH --job-name=safedrive-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
bash scripts/eval.sh "$@"
