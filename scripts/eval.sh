#!/usr/bin/env bash
# Closed-loop evaluation in CARLA 0.9.15 (one CARLA server per GPU).
#
#   bash scripts/eval.sh <benchmark> <checkpoint> [options]
#
#   benchmark:    bench2drive | carla_f | b2d_c | b2d_adv
#   checkpoint:   a released model (<model>/<checkpoint>, config.yaml next to it) or a
#                 training checkpoint work_dirs/safedrive_vla/<experiment>/<timestamp>/checkpoints/epoch=013.ckpt
#   --gpu-ids 0,1,...           GPUs to use (default: all visible GPUs)
#   --nav-signal NAME           Bench2Drive navigation signal: target_point (default) | command
#   --output-dir DIR            default: work_dirs/eval/<benchmark>/<experiment>
#   --save-viz                  save front-camera images with the predictions
#
# Requires CARLA_ROOT. Interrupted runs resume: completed routes are skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BENCHMARK="${1:?usage: $0 <benchmark> <checkpoint> [options]}"
CHECKPOINT="${2:?usage: $0 <benchmark> <checkpoint> [options]}"
shift 2
GPU_IDS=""
OUTPUT_DIR=""
export SAFEDRIVE_NAV_SIGNAL="target_point"
export SAFEDRIVE_SAVE_VIZ="0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-ids)    GPU_IDS="$2"; shift 2 ;;
    --nav-signal) SAFEDRIVE_NAV_SIGNAL="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --save-viz)   SAFEDRIVE_SAVE_VIZ="1"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

case "${BENCHMARK}" in
  bench2drive) AGENT=team_code/agent_safedrive.py;          NUM_ROUTES=220 ;;
  carla_f)     AGENT=team_code/agent_safedrive_carla_f.py;  NUM_ROUTES=210 ;;
  b2d_c)       AGENT=team_code/agent_safedrive_b2d_c.py;    NUM_ROUTES=150 ;;
  b2d_adv)     AGENT=team_code/agent_safedrive_b2d_c.py;    NUM_ROUTES=15 ;;
  *) echo "Unknown benchmark: ${BENCHMARK}" >&2; exit 1 ;;
esac
: "${CARLA_ROOT:?set CARLA_ROOT to the CARLA 0.9.15 installation}"
[[ -e "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
if [[ -z "${GPU_IDS}" ]]; then
  GPU_IDS="$(seq -s, 0 $(( $(nvidia-smi -L | grep -c '^GPU ') - 1 )))"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  CKPT_DIR="$(dirname "$(realpath "${CHECKPOINT}")")"
  if [[ -f "${CKPT_DIR}/config.yaml" ]]; then
    EXPERIMENT="$(basename "${CKPT_DIR}")"                            # released model
  else
    EXPERIMENT="$(basename "$(dirname "$(dirname "${CKPT_DIR}")")")"  # training run
  fi
  OUTPUT_DIR="work_dirs/eval/${BENCHMARK}/${EXPERIMENT}"
  [[ "${BENCHMARK}" == bench2drive && "${SAFEDRIVE_NAV_SIGNAL}" == command ]] && OUTPUT_DIR="${OUTPUT_DIR}_command"
fi
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

python tools/run_distributed_eval.py \
  --routes-dir "benchmark/data/${BENCHMARK}" \
  --output-dir "${OUTPUT_DIR}" \
  --agent "${AGENT}" \
  --agent-config "$(realpath "${CHECKPOINT}")" \
  --gpu-ids "${GPU_IDS}"

if [[ "${BENCHMARK}" == carla_f ]]; then
  python benchmark/metrics/carla_f_metrics.py "${OUTPUT_DIR}" --routes-dir "benchmark/data/${BENCHMARK}"
else
  python benchmark/metrics/bench2drive_metrics.py "${OUTPUT_DIR}" --num-routes "${NUM_ROUTES}"
fi
