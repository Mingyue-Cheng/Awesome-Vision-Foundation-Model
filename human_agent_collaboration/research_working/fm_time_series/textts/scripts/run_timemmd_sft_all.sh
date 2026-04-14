#!/usr/bin/env bash
set -euo pipefail

TEXTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$TEXTTS_DIR/.." && pwd)"
source "$TEXTTS_DIR/scripts/_timemmd_batch_common.sh"

TIMEMMD_ROOT="${TIMEMMD_ROOT:-$PROJECT_ROOT/Time-MMD}"
BATCH_OUTPUT_DIR="${BATCH_OUTPUT_DIR:-$PROJECT_ROOT/outputs/timemmd_sft_all}"
SUMMARY_FILE="${SUMMARY_FILE:-$BATCH_OUTPUT_DIR/summary.tsv}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

mkdir -p "$BATCH_OUTPUT_DIR"
printf 'domain\tstatus\ttrain_loss\ttrain_steps\toutput_dir\tlog_path\n' > "$SUMMARY_FILE"

mapfile -t DOMAIN_LIST < <(resolve_timemmd_domains "$TIMEMMD_ROOT")
printf 'Running batch SFT for %s domains\n' "${#DOMAIN_LIST[@]}"

for domain in "${DOMAIN_LIST[@]}"; do
  output_dir="$BATCH_OUTPUT_DIR/$domain"
  log_path="$output_dir/run.log"
  mkdir -p "$output_dir"

  printf '\n[%s] sft -> %s\n' "$domain" "$output_dir"
  if TIMEMMD_ROOT="$TIMEMMD_ROOT" OUTPUT_DIR="$output_dir" "$TEXTTS_DIR/scripts/run_timemmd_sft.sh" "$domain" >"$log_path" 2>&1; then
    train_loss="$(json_field_or_na "$output_dir/metadata.json" "train_metrics.loss")"
    train_steps="$(json_field_or_na "$output_dir/metadata.json" "train_metrics.steps")"
    printf '%s\tsuccess\t%s\t%s\t%s\t%s\n' "$domain" "$train_loss" "$train_steps" "$output_dir" "$log_path" >> "$SUMMARY_FILE"
  else
    printf '%s\tfailed\tNA\tNA\t%s\t%s\n' "$domain" "$output_dir" "$log_path" >> "$SUMMARY_FILE"
    print_failure_tail "$log_path"
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit 1
    fi
  fi
done

printf '\nBatch SFT summary written to %s\n' "$SUMMARY_FILE"
