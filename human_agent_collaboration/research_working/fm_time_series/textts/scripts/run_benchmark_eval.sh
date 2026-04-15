#!/usr/bin/env bash
set -euo pipefail

TEXTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$TEXTTS_DIR/.." && pwd)"

# Unified benchmark evaluation entrypoint.
# Supported modes:
# - MODE=gift : run GIFT-Eval only
# - MODE=ltsf : run classic LTSF benchmark only
# - MODE=all  : run both benchmark suites sequentially
# - MODE=auto : infer from env vars

MODE="${MODE:-auto}"
RUN_LABEL="${RUN_LABEL:-benchmark}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
TEXTTS_MODULES_PATH="${TEXTTS_MODULES_PATH:-}"
MODEL_NAME="${MODEL_NAME:-}"
DEVICE="${DEVICE:-}"
TORCH_DTYPE="${TORCH_DTYPE:-}"
DEVICE_MAP="${DEVICE_MAP:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-}"
REGIME="${REGIME:-}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"
TRAIN_DATASETS="${TRAIN_DATASETS:-}"
ENFORCE_PROTOCOL="${ENFORCE_PROTOCOL:-}"

GIFT_SOURCE="${GIFT_SOURCE:-}"
GIFT_CONFIG="${GIFT_CONFIG:-}"
LTSF_ROOT="${LTSF_ROOT:-}"
DATASETS="${DATASETS:-}"

GIFT_OUTPUT_DIR="${GIFT_OUTPUT_DIR:-$PROJECT_ROOT/outputs/${RUN_LABEL}_gift}"
LTSF_OUTPUT_DIR="${LTSF_OUTPUT_DIR:-$PROJECT_ROOT/outputs/${RUN_LABEL}_ltsf}"

infer_mode() {
  if [[ -n "$MODE" && "$MODE" != "auto" ]]; then
    printf '%s\n' "$MODE"
    return
  fi

  if [[ -n "$GIFT_SOURCE" && -n "$LTSF_ROOT" ]]; then
    printf 'all\n'
    return
  fi
  if [[ -n "$GIFT_SOURCE" ]]; then
    printf 'gift\n'
    return
  fi
  if [[ -n "$LTSF_ROOT" ]]; then
    printf 'ltsf\n'
    return
  fi
  printf 'all\n'
}

export_common_env() {
  if [[ -n "$CHECKPOINT_DIR" ]]; then
    export CHECKPOINT_DIR
  fi
  if [[ -n "$TEXTTS_MODULES_PATH" ]]; then
    export TEXTTS_MODULES_PATH
  fi
  if [[ -n "$MODEL_NAME" ]]; then
    export MODEL_NAME
  fi
  if [[ -n "$DEVICE" ]]; then
    export DEVICE
  fi
  if [[ -n "$TORCH_DTYPE" ]]; then
    export TORCH_DTYPE
  fi
  if [[ -n "$DEVICE_MAP" ]]; then
    export DEVICE_MAP
  fi
  if [[ -n "$LOCAL_FILES_ONLY" ]]; then
    export LOCAL_FILES_ONLY
  fi
  if [[ -n "$REGIME" ]]; then
    export REGIME
  fi
  if [[ -n "$TRAIN_MANIFEST" ]]; then
    export TRAIN_MANIFEST
  fi
  if [[ -n "$TRAIN_DATASETS" ]]; then
    export TRAIN_DATASETS
  fi
  if [[ -n "$ENFORCE_PROTOCOL" ]]; then
    export ENFORCE_PROTOCOL
  fi
}

run_gift() {
  export_common_env
  export OUTPUT_DIR="$GIFT_OUTPUT_DIR"
  if [[ -n "$GIFT_SOURCE" ]]; then
    export GIFT_SOURCE
  fi
  if [[ -n "$GIFT_CONFIG" ]]; then
    export GIFT_CONFIG
  fi
  printf 'Unified benchmark eval: mode=gift output=%s source=%s\n' "$OUTPUT_DIR" "${GIFT_SOURCE:-Salesforce/GiftEval}"
  bash "$TEXTTS_DIR/scripts/run_gift_eval.sh"
}

run_ltsf() {
  export_common_env
  export OUTPUT_DIR="$LTSF_OUTPUT_DIR"
  if [[ -n "$LTSF_ROOT" ]]; then
    export LTSF_ROOT
  fi
  if [[ -n "$DATASETS" ]]; then
    export DATASETS
  fi
  printf 'Unified benchmark eval: mode=ltsf output=%s root=%s\n' "$OUTPUT_DIR" "${LTSF_ROOT:-$PROJECT_ROOT/数据集/LTSF}"
  bash "$TEXTTS_DIR/scripts/run_ltsf_eval.sh"
}

MODE="$(infer_mode)"

case "$MODE" in
  gift)
    run_gift
    ;;
  ltsf)
    run_ltsf
    ;;
  all)
    run_gift
    run_ltsf
    ;;
  *)
    echo "Unknown MODE=$MODE. Expected one of: auto, gift, ltsf, all." >&2
    exit 1
    ;;
esac
