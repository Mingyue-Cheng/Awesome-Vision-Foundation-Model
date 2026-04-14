#!/usr/bin/env bash
set -euo pipefail

TEXTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$TEXTTS_DIR/.." && pwd)"

DOMAINS="${DOMAINS:-}"
DOMAIN="${1:-${DOMAIN:-}}"
if [[ -z "$DOMAINS" && -z "$DOMAIN" ]]; then
  DOMAIN="Energy"
fi
TIMEMMD_ROOT="${TIMEMMD_ROOT:-$PROJECT_ROOT/Time-MMD}"
TARGET_COL="${TARGET_COL:-OT}"
LOOKBACK="${LOOKBACK:-32}"
HORIZON="${HORIZON:-8}"
STRIDE="${STRIDE:-16}"
MAX_WINDOWS="${MAX_WINDOWS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEPS="${STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}"
TORCH_DTYPE="${TORCH_DTYPE:-}"
DEVICE_MAP="${DEVICE_MAP:-}"
USE_LORA="${USE_LORA:-0}"
DEVICE="${DEVICE:-cpu}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
TEXTTS_MODULES_PATH="${TEXTTS_MODULES_PATH:-}"
USE_FIXED_SPLITS="${USE_FIXED_SPLITS:-1}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
SPLIT_VAL_RATIO="${SPLIT_VAL_RATIO:-0.1}"
SPLIT_TEST_RATIO="${SPLIT_TEST_RATIO:-0.1}"
EVAL_RATIO="${EVAL_RATIO:-0.0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-2}"
EVAL_POINT_STRATEGY="${EVAL_POINT_STRATEGY:-greedy}"
EVAL_NUM_PROB_SAMPLES="${EVAL_NUM_PROB_SAMPLES:-0}"
EVAL_PROB_TEMPERATURE="${EVAL_PROB_TEMPERATURE:-1.0}"
EVAL_PROB_TOP_P="${EVAL_PROB_TOP_P:-0.9}"
RUN_LABEL="${RUN_LABEL:-${DOMAIN:-joint}}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/timemmd_sft_${RUN_LABEL}}"

CMD=(
  python3 -m textts.training.sft
  --data-source timemmd
  --timemmd-root "$TIMEMMD_ROOT"
  --target-col "$TARGET_COL"
  --lookback "$LOOKBACK"
  --horizon "$HORIZON"
  --stride "$STRIDE"
  --max-windows "$MAX_WINDOWS"
  --batch-size "$BATCH_SIZE"
  --steps "$STEPS"
  --learning-rate "$LEARNING_RATE"
  --model-name "$MODEL_NAME"
  --device "$DEVICE"
  --eval-ratio "$EVAL_RATIO"
  --eval-max-samples "$EVAL_MAX_SAMPLES"
  --eval-point-strategy "$EVAL_POINT_STRATEGY"
  --eval-num-prob-samples "$EVAL_NUM_PROB_SAMPLES"
  --eval-prob-temperature "$EVAL_PROB_TEMPERATURE"
  --eval-prob-top-p "$EVAL_PROB_TOP_P"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$DOMAINS" ]]; then
  CMD+=(--domains "$DOMAINS")
else
  CMD+=(--domain "$DOMAIN")
fi

if [[ "$USE_LORA" == "1" ]]; then
  CMD+=(--use-lora)
fi

if [[ -n "$TORCH_DTYPE" ]]; then
  CMD+=(--torch-dtype "$TORCH_DTYPE")
fi

if [[ -n "$DEVICE_MAP" ]]; then
  CMD+=(--device-map "$DEVICE_MAP")
fi

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  CMD+=(--local-files-only)
fi

if [[ -n "$CHECKPOINT_DIR" ]]; then
  CMD+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

if [[ -n "$TEXTTS_MODULES_PATH" ]]; then
  CMD+=(--textts-modules-path "$TEXTTS_MODULES_PATH")
fi

if [[ "$USE_FIXED_SPLITS" == "1" ]]; then
  CMD+=(--use-fixed-splits --train-split "$TRAIN_SPLIT" --eval-split "$EVAL_SPLIT" --val-ratio "$SPLIT_VAL_RATIO" --test-ratio "$SPLIT_TEST_RATIO")
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf 'Running Time-MMD SFT with domains=%s root=%s init=%s output=%s\n' "${DOMAINS:-$DOMAIN}" "$TIMEMMD_ROOT" "${CHECKPOINT_DIR:-$MODEL_NAME}" "$OUTPUT_DIR"
"${CMD[@]}"
