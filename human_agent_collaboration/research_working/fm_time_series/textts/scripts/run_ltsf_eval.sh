#!/usr/bin/env bash
set -euo pipefail

TEXTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$TEXTTS_DIR/.." && pwd)"

LTSF_ROOT="${LTSF_ROOT:-$PROJECT_ROOT/数据集/LTSF}"
DATASETS="${DATASETS:-all}"
HORIZONS="${HORIZONS:-96,192,336,720}"
LOOKBACK="${LOOKBACK:-96}"
SPLIT="${SPLIT:-test}"
STRIDE="${STRIDE:-1}"
MAX_WINDOWS="${MAX_WINDOWS:-}"
TARGET_MODE="${TARGET_MODE:-all}"
TARGET_COLS="${TARGET_COLS:-}"
MAX_TARGETS="${MAX_TARGETS:-}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.2}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
TEXTTS_MODULES_PATH="${TEXTTS_MODULES_PATH:-}"
TORCH_DTYPE="${TORCH_DTYPE:-}"
DEVICE_MAP="${DEVICE_MAP:-}"
DEVICE="${DEVICE:-cpu}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
POINT_STRATEGY="${POINT_STRATEGY:-greedy}"
POINT_TEMPERATURE="${POINT_TEMPERATURE:-1.0}"
POINT_TOP_P="${POINT_TOP_P:-1.0}"
NUM_PROB_SAMPLES="${NUM_PROB_SAMPLES:-0}"
PROB_TEMPERATURE="${PROB_TEMPERATURE:-1.0}"
PROB_TOP_P="${PROB_TOP_P:-0.9}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
REGIME="${REGIME:-auto}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"
TRAIN_DATASETS="${TRAIN_DATASETS:-}"
ENFORCE_PROTOCOL="${ENFORCE_PROTOCOL:-0}"
RUN_LABEL="${RUN_LABEL:-${DATASETS//,/+}_${SPLIT}}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/ltsf_eval_${RUN_LABEL}}"

CMD=(
  python3 -m textts.eval.ltsf_eval
  --ltsf-root "$LTSF_ROOT"
  --datasets "$DATASETS"
  --horizons "$HORIZONS"
  --lookback "$LOOKBACK"
  --split "$SPLIT"
  --stride "$STRIDE"
  --target-mode "$TARGET_MODE"
  --val-ratio "$VAL_RATIO"
  --test-ratio "$TEST_RATIO"
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

if [[ -n "$MAX_WINDOWS" ]]; then
  CMD+=(--max-windows "$MAX_WINDOWS")
fi

if [[ -n "$TARGET_COLS" ]]; then
  CMD+=(--target-cols "$TARGET_COLS")
fi

if [[ -n "$MAX_TARGETS" ]]; then
  CMD+=(--max-targets "$MAX_TARGETS")
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

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf 'Running LTSF eval with datasets=%s horizons=%s root=%s output=%s\n' "$DATASETS" "$HORIZONS" "$LTSF_ROOT" "$OUTPUT_DIR"
"${CMD[@]}"
