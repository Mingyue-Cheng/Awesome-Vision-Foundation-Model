#!/usr/bin/env bash
set -euo pipefail

TEXTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOMAINS="${DOMAINS:-Energy Climate Traffic}"
RUN_LABEL="${RUN_LABEL:-joint}"

export DOMAINS
export RUN_LABEL

"$TEXTTS_DIR/scripts/run_timemmd_pretrain.sh"
