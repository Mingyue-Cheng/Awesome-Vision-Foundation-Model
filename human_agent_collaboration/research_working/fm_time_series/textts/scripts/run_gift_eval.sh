#!/usr/bin/env bash
set -euo pipefail

TEXTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$TEXTTS_DIR/.." && pwd)"

GIFT_SOURCE="${GIFT_SOURCE:-Salesforce/GiftEval}"
GIFT_CONFIG="${GIFT_CONFIG:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
PROTOCOL="${PROTOCOL:-zero-shot}"
DATASET_FILTER="${DATASET_FILTER:-}"
MAX_TRAIN_RECORDS="${MAX_TRAIN_RECORDS:-}"
MAX_EVAL_RECORDS="${MAX_EVAL_RECORDS:-}"
FEW_SHOT_RATIO="${FEW_SHOT_RATIO:-0.05}"
FEW_SHOT_STEPS="${FEW_SHOT_STEPS:-20}"
FEW_SHOT_BATCH_SIZE="${FEW_SHOT_BATCH_SIZE:-4}"
FEW_SHOT_LEARNING_RATE="${FEW_SHOT_LEARNING_RATE:-1e-4}"
FEW_SHOT_USE_LORA="${FEW_SHOT_USE_LORA:-0}"
SFT_CONTEXT_MODE="${SFT_CONTEXT_MODE:-mixed}"
SFT_CONTEXT_CACHE="${SFT_CONTEXT_CACHE:-}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
TEXTTS_MODULES_PATH="${TEXTTS_MODULES_PATH:-}"
TORCH_DTYPE="${TORCH_DTYPE:-}"
DEVICE_MAP="${DEVICE_MAP:-}"
DEVICE="${DEVICE:-cpu}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
POINT_STRATEGY="${POINT_STRATEGY:-greedy}"
POINT_TEMPERATURE="${POINT_TEMPERATURE:-1.0}"
POINT_TOP_P="${POINT_TOP_P:-1.0}"
NUM_PROB_SAMPLES="${NUM_PROB_SAMPLES:-16}"
PROB_TEMPERATURE="${PROB_TEMPERATURE:-1.0}"
PROB_TOP_P="${PROB_TOP_P:-0.9}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
REGIME="${REGIME:-auto}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"
TRAIN_DATASETS="${TRAIN_DATASETS:-}"
ENFORCE_PROTOCOL="${ENFORCE_PROTOCOL:-0}"
SAVE_FEW_SHOT_CHECKPOINT="${SAVE_FEW_SHOT_CHECKPOINT:-0}"
RUN_LABEL="${RUN_LABEL:-gift_${PROTOCOL}}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/${RUN_LABEL}}"

CMD=(
  python3 -m textts.eval.gift_eval
  --gift-source "$GIFT_SOURCE"
  --train-split "$TRAIN_SPLIT"
  --eval-split "$EVAL_SPLIT"
  --protocol "$PROTOCOL"
  --few-shot-ratio "$FEW_SHOT_RATIO"
  --few-shot-steps "$FEW_SHOT_STEPS"
  --few-shot-batch-size "$FEW_SHOT_BATCH_SIZE"
  --few-shot-learning-rate "$FEW_SHOT_LEARNING_RATE"
  --sft-context-mode "$SFT_CONTEXT_MODE"
  --model-name "$MODEL_NAME"
  --device "$DEVICE"
  --point-strategy "$POINT_STRATEGY"
  --point-temperature "$POINT_TEMPERATURE"
  --point-top-p "$POINT_TOP_P"
  --num-prob-samples "$NUM_PROB_SAMPLES"
  --prob-temperature "$PROB_TEMPERATURE"
  --prob-top-p "$PROB_TOP_P"
  --regime "$REGIME"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$GIFT_CONFIG" ]]; then
  CMD+=(--gift-config "$GIFT_CONFIG")
fi

if [[ -n "$DATASET_FILTER" ]]; then
  CMD+=(--dataset-filter "$DATASET_FILTER")
fi

if [[ -n "$MAX_TRAIN_RECORDS" ]]; then
  CMD+=(--max-train-records "$MAX_TRAIN_RECORDS")
fi

if [[ -n "$MAX_EVAL_RECORDS" ]]; then
  CMD+=(--max-eval-records "$MAX_EVAL_RECORDS")
fi

if [[ "$FEW_SHOT_USE_LORA" == "1" ]]; then
  CMD+=(--few-shot-use-lora)
fi

if [[ -n "$SFT_CONTEXT_CACHE" ]]; then
  CMD+=(--sft-context-cache "$SFT_CONTEXT_CACHE")
fi

if [[ -n "$CHECKPOINT_DIR" ]]; then
  CMD+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

if [[ -n "$TEXTTS_MODULES_PATH" ]]; then
  CMD+=(--textts-modules-path "$TEXTTS_MODULES_PATH")
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

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ -n "$TRAIN_MANIFEST" ]]; then
  CMD+=(--train-manifest "$TRAIN_MANIFEST")
fi

if [[ -n "$TRAIN_DATASETS" ]]; then
  CMD+=(--train-datasets "$TRAIN_DATASETS")
fi

if [[ "$ENFORCE_PROTOCOL" == "1" ]]; then
  CMD+=(--enforce-protocol)
fi

if [[ "$SAVE_FEW_SHOT_CHECKPOINT" == "1" ]]; then
  CMD+=(--save-few-shot-checkpoint)
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf 'Running GIFT eval with protocol=%s source=%s output=%s\n' "$PROTOCOL" "$GIFT_SOURCE" "$OUTPUT_DIR"
"${CMD[@]}"
