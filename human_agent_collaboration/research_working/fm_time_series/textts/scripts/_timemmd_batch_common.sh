#!/usr/bin/env bash

resolve_timemmd_domains() {
  local timemmd_root="$1"

  if [[ -n "${DOMAINS:-}" ]]; then
    python3 - "$DOMAINS" <<'PY'
import re
import sys

parts = [part for part in re.split(r"[\s,]+", sys.argv[1].strip()) if part]
for part in parts:
    print(part)
PY
    return
  fi

  if [[ -d "$timemmd_root/numerical" ]]; then
    find "$timemmd_root/numerical" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | LC_ALL=C sort
    return
  fi

  printf '%s\n' \
    Agriculture \
    Climate \
    Economy \
    Energy \
    Environment \
    Health_AFR \
    Health_US \
    Security \
    SocialGood \
    Traffic
}


json_field_or_na() {
  local json_path="$1"
  local field_path="$2"

  python3 - "$json_path" "$field_path" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
field_path = sys.argv[2]

if not path.exists():
    print("NA")
    raise SystemExit

try:
    value = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("NA")
    raise SystemExit

for part in field_path.split("."):
    if isinstance(value, dict) and part in value:
        value = value[part]
    else:
        print("NA")
        raise SystemExit

if value is None:
    print("NA")
else:
    print(value)
PY
}


print_failure_tail() {
  local log_path="$1"
  local tail_lines="${TAIL_LINES_ON_FAILURE:-40}"

  if [[ -f "$log_path" ]]; then
    printf '\nLast %s log lines from %s:\n' "$tail_lines" "$log_path" >&2
    tail -n "$tail_lines" "$log_path" >&2 || true
  fi
}
