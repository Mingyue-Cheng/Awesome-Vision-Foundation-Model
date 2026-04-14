#!/usr/bin/env bash
set -euo pipefail

TEXTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$TEXTTS_DIR/.." && pwd)"
source "$TEXTTS_DIR/scripts/_timemmd_batch_common.sh"

TIMEMMD_ROOT="${TIMEMMD_ROOT:-$PROJECT_ROOT/Time-MMD}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$PROJECT_ROOT/outputs/timemmd_pretrain_all}"
BATCH_OUTPUT_DIR="${BATCH_OUTPUT_DIR:-$PROJECT_ROOT/outputs/timemmd_eval_all}"
SUMMARY_FILE="${SUMMARY_FILE:-$BATCH_OUTPUT_DIR/summary.tsv}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

mkdir -p "$BATCH_OUTPUT_DIR"
printf 'domain\tstatus\tmae\trmse\tcrps\tcoverage_80\tinterval_width_80\toutput_dir\tlog_path\n' > "$SUMMARY_FILE"

mapfile -t DOMAIN_LIST < <(resolve_timemmd_domains "$TIMEMMD_ROOT")
printf 'Running batch eval for %s domains\n' "${#DOMAIN_LIST[@]}"

for domain in "${DOMAIN_LIST[@]}"; do
  output_dir="$BATCH_OUTPUT_DIR/$domain"
  log_path="$output_dir/run.log"
  checkpoint_dir="$CHECKPOINT_BASE_DIR/$domain"
  mkdir -p "$output_dir"

  printf '\n[%s] eval -> %s\n' "$domain" "$output_dir"
  if TIMEMMD_ROOT="$TIMEMMD_ROOT" OUTPUT_DIR="$output_dir" CHECKPOINT_DIR="$checkpoint_dir" "$TEXTTS_DIR/scripts/run_timemmd_eval.sh" "$domain" >"$log_path" 2>&1; then
    mae="$(json_field_or_na "$output_dir/metrics.json" "mae")"
    rmse="$(json_field_or_na "$output_dir/metrics.json" "rmse")"
    crps="$(json_field_or_na "$output_dir/metrics.json" "crps")"
    coverage="$(json_field_or_na "$output_dir/metrics.json" "coverage_80")"
    width="$(json_field_or_na "$output_dir/metrics.json" "interval_width_80")"
    printf '%s\tsuccess\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$domain" "$mae" "$rmse" "$crps" "$coverage" "$width" "$output_dir" "$log_path" >> "$SUMMARY_FILE"
  else
    printf '%s\tfailed\tNA\tNA\tNA\tNA\tNA\t%s\t%s\n' "$domain" "$output_dir" "$log_path" >> "$SUMMARY_FILE"
    print_failure_tail "$log_path"
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit 1
    fi
  fi
done

printf '\nBatch eval summary written to %s\n' "$SUMMARY_FILE"
