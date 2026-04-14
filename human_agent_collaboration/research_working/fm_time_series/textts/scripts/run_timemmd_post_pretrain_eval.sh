#!/usr/bin/env bash
set -euo pipefail

TEXTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Unified post-pretrain evaluation entrypoint.
# Supported modes:
# - MODE=single : one checkpoint, one domain
# - MODE=joint  : one shared checkpoint, multiple domains in one eval run
# - MODE=batch  : one checkpoint directory per domain, sequential batch eval
# - MODE=auto   : infer from env vars

MODE="${MODE:-auto}"
DOMAINS="${DOMAINS:-}"
DOMAIN="${1:-${DOMAIN:-}}"
RUN_LABEL="${RUN_LABEL:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-}"
BATCH_OUTPUT_DIR="${BATCH_OUTPUT_DIR:-}"

infer_mode() {
  if [[ -n "$MODE" && "$MODE" != "auto" ]]; then
    printf '%s\n' "$MODE"
    return
  fi
  if [[ -n "$CHECKPOINT_BASE_DIR" ]]; then
    printf 'batch\n'
    return
  fi
  if [[ -n "$DOMAINS" ]]; then
    printf 'joint\n'
    return
  fi
  printf 'single\n'
}

MODE="$(infer_mode)"

case "$MODE" in
  single)
    if [[ -z "$DOMAIN" ]]; then
      DOMAIN="Energy"
    fi
    if [[ -n "$RUN_LABEL" ]]; then
      export RUN_LABEL
    fi
    if [[ -n "$CHECKPOINT_DIR" ]]; then
      export CHECKPOINT_DIR
    fi
    printf 'Unified post-pretrain eval: mode=single domain=%s checkpoint=%s\n' "$DOMAIN" "${CHECKPOINT_DIR:-auto}"
    exec "$TEXTTS_DIR/scripts/run_timemmd_eval.sh" "$DOMAIN"
    ;;
  joint)
    if [[ -z "$DOMAINS" ]]; then
      echo "MODE=joint requires DOMAINS." >&2
      exit 1
    fi
    if [[ -z "$RUN_LABEL" ]]; then
      RUN_LABEL="joint"
    fi
    export DOMAINS
    export RUN_LABEL
    if [[ -n "$CHECKPOINT_DIR" ]]; then
      export CHECKPOINT_DIR
    fi
    printf 'Unified post-pretrain eval: mode=joint domains=%s checkpoint=%s\n' "$DOMAINS" "${CHECKPOINT_DIR:-auto}"
    exec "$TEXTTS_DIR/scripts/run_timemmd_eval.sh"
    ;;
  batch)
    if [[ -n "$DOMAINS" ]]; then
      export DOMAINS
    fi
    if [[ -n "$CHECKPOINT_BASE_DIR" ]]; then
      export CHECKPOINT_BASE_DIR
    fi
    if [[ -n "$BATCH_OUTPUT_DIR" ]]; then
      export BATCH_OUTPUT_DIR
    fi
    printf 'Unified post-pretrain eval: mode=batch checkpoint_base=%s output=%s\n' "${CHECKPOINT_BASE_DIR:-auto}" "${BATCH_OUTPUT_DIR:-auto}"
    exec "$TEXTTS_DIR/scripts/run_timemmd_eval_all.sh"
    ;;
  *)
    echo "Unknown MODE=$MODE. Expected one of: auto, single, joint, batch." >&2
    exit 1
    ;;
esac
