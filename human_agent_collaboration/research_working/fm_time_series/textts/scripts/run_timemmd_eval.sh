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
SPLIT="${SPLIT:-test}"
SPLIT_VAL_RATIO="${SPLIT_VAL_RATIO:-0.1}"
SPLIT_TEST_RATIO="${SPLIT_TEST_RATIO:-0.1}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}"
RUN_LABEL="${RUN_LABEL:-${DOMAIN:-joint}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/outputs/timemmd_pretrain_${RUN_LABEL}}"
TEXTTS_MODULES_PATH="${TEXTTS_MODULES_PATH:-}"
TORCH_DTYPE="${TORCH_DTYPE:-}"
DEVICE_MAP="${DEVICE_MAP:-}"
DEVICE="${DEVICE:-cpu}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
POINT_STRATEGY="${POINT_STRATEGY:-greedy}"
POINT_TEMPERATURE="${POINT_TEMPERATURE:-1.0}"
POINT_TOP_P="${POINT_TOP_P:-1.0}"
NUM_PROB_SAMPLES="${NUM_PROB_SAMPLES:-4}"
PROB_TEMPERATURE="${PROB_TEMPERATURE:-1.0}"
PROB_TOP_P="${PROB_TOP_P:-0.9}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/timemmd_eval_${RUN_LABEL}}"

CMD=(
  python3 -m textts.eval.forecast_eval
  --data-source timemmd
  --timemmd-root "$TIMEMMD_ROOT"
  --target-col "$TARGET_COL"
  --lookback "$LOOKBACK"
  --horizon "$HORIZON"
  --stride "$STRIDE"
  --max-windows "$MAX_WINDOWS"
  --split "$SPLIT"
  --val-ratio "$SPLIT_VAL_RATIO"
  --test-ratio "$SPLIT_TEST_RATIO"
  --max-samples "$MAX_SAMPLES"
  --model-name "$MODEL_NAME"
  --device "$DEVICE"
  --point-strategy "$POINT_STRATEGY"
  --point-temperature "$POINT_TEMPERATURE"
  --point-top-p "$POINT_TOP_P"
  --num-prob-samples "$NUM_PROB_SAMPLES"
  --prob-temperature "$PROB_TEMPERATURE"
  --prob-top-p "$PROB_TOP_P"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$DOMAINS" ]]; then
  CMD+=(--domains "$DOMAINS")
else
  CMD+=(--domain "$DOMAIN")
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

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf 'Running Time-MMD eval with domains=%s root=%s output=%s\n' "${DOMAINS:-$DOMAIN}" "$TIMEMMD_ROOT" "$OUTPUT_DIR"
"${CMD[@]}"
