#!/usr/bin/env bash
set -euo pipefail

TEXTTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$TEXTTS_DIR/.." && pwd)"
source "$TEXTTS_DIR/scripts/_timemmd_batch_common.sh"

default_timemmd_root() {
  if [[ -d "$PROJECT_ROOT/数据集/Time-MMD" ]]; then
    printf '%s\n' "$PROJECT_ROOT/数据集/Time-MMD"
    return
  fi
  printf '%s\n' "$PROJECT_ROOT/Time-MMD"
}

TIMEMMD_ROOT="${TIMEMMD_ROOT:-$(default_timemmd_root)}"
EXPERIMENT_NAME="timemmd_experiment_1"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/$EXPERIMENT_NAME"
MANIFEST_PATH="$OUTPUT_ROOT/manifest.json"
DOMAIN_SELECTION_PATH="$OUTPUT_ROOT/domain_selection.json"
SUMMARY_JSON="$OUTPUT_ROOT/summary.json"
SUMMARY_TSV="$OUTPUT_ROOT/summary.tsv"

TARGET_COL="${TARGET_COL:-OT}"
LOOKBACK="${LOOKBACK:-32}"
HORIZON="${HORIZON:-8}"
STRIDE="${STRIDE:-16}"
MAX_WINDOWS="${MAX_WINDOWS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}"
TORCH_DTYPE="${TORCH_DTYPE:-}"
DEVICE_MAP="${DEVICE_MAP:-}"
DEVICE="${DEVICE:-cpu}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

PRETRAIN_STEPS="${PRETRAIN_STEPS:-1}"
PRETRAIN_LEARNING_RATE="${PRETRAIN_LEARNING_RATE:-1e-4}"
PRETRAIN_PRED_PROBABILITY="${PRETRAIN_PRED_PROBABILITY:-0.7}"
PRETRAIN_EVAL_RATIO="${PRETRAIN_EVAL_RATIO:-0.0}"
PRETRAIN_EVAL_MAX_SAMPLES="${PRETRAIN_EVAL_MAX_SAMPLES:-2}"

SFT_STEPS="${SFT_STEPS:-1}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-4}"
SFT_USE_LORA="${SFT_USE_LORA:-0}"
SFT_EVAL_RATIO="${SFT_EVAL_RATIO:-0.0}"
SFT_EVAL_MAX_SAMPLES="${SFT_EVAL_MAX_SAMPLES:-2}"

POST_PRETRAIN_MAX_SAMPLES="${POST_PRETRAIN_MAX_SAMPLES:-2}"
POST_PRETRAIN_POINT_STRATEGY="${POST_PRETRAIN_POINT_STRATEGY:-greedy}"
POST_PRETRAIN_POINT_TEMPERATURE="${POST_PRETRAIN_POINT_TEMPERATURE:-1.0}"
POST_PRETRAIN_POINT_TOP_P="${POST_PRETRAIN_POINT_TOP_P:-1.0}"
POST_PRETRAIN_NUM_PROB_SAMPLES="${POST_PRETRAIN_NUM_PROB_SAMPLES:-4}"
POST_PRETRAIN_PROB_TEMPERATURE="${POST_PRETRAIN_PROB_TEMPERATURE:-1.0}"
POST_PRETRAIN_PROB_TOP_P="${POST_PRETRAIN_PROB_TOP_P:-0.9}"

POST_SFT_MAX_SAMPLES="${POST_SFT_MAX_SAMPLES:-2}"
POST_SFT_POINT_STRATEGY="${POST_SFT_POINT_STRATEGY:-greedy}"
POST_SFT_POINT_TEMPERATURE="${POST_SFT_POINT_TEMPERATURE:-1.0}"
POST_SFT_POINT_TOP_P="${POST_SFT_POINT_TOP_P:-1.0}"
POST_SFT_NUM_PROB_SAMPLES="${POST_SFT_NUM_PROB_SAMPLES:-4}"
POST_SFT_PROB_TEMPERATURE="${POST_SFT_PROB_TEMPERATURE:-1.0}"
POST_SFT_PROB_TOP_P="${POST_SFT_PROB_TOP_P:-0.9}"

if [[ -n "${DOMAIN:-}" || -n "${DOMAINS:-}" ]]; then
  printf 'Ignoring DOMAIN/DOMAINS overrides. %s selects domains via preflight.\n' "$EXPERIMENT_NAME"
fi

unset DOMAIN DOMAINS RUN_LABEL BASE_RUN_LABEL PRETRAIN_RUN_LABEL POST_PRETRAIN_EVAL_RUN_LABEL SFT_RUN_LABEL POST_SFT_EVAL_RUN_LABEL || true
unset PRETRAIN_OUTPUT_DIR POST_PRETRAIN_EVAL_OUTPUT_DIR SFT_OUTPUT_DIR POST_SFT_EVAL_OUTPUT_DIR || true

if [[ ! -d "$TIMEMMD_ROOT" ]]; then
  echo "Time-MMD root not found: $TIMEMMD_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

DOMAIN_CANDIDATES=()
while IFS= read -r domain_name; do
  if [[ -n "$domain_name" ]]; then
    DOMAIN_CANDIDATES+=("$domain_name")
  fi
done < <(resolve_timemmd_domains "$TIMEMMD_ROOT")

if [[ "${#DOMAIN_CANDIDATES[@]}" -eq 0 ]]; then
  echo "No Time-MMD domain candidates found under $TIMEMMD_ROOT" >&2
  exit 1
fi

DOMAIN_CANDIDATE_TEXT="${DOMAIN_CANDIDATES[*]}"
python3 - "$DOMAIN_SELECTION_PATH" "$TIMEMMD_ROOT" "$DOMAIN_CANDIDATE_TEXT" "$TARGET_COL" "$LOOKBACK" "$HORIZON" "$STRIDE" <<'PY'
import json
import pathlib
import re
import sys

from textts.data.timemmd_loader import TimeMMDWindowConfig, load_timemmd_windows

output_path = pathlib.Path(sys.argv[1])
timemmd_root = pathlib.Path(sys.argv[2])
candidates = [part for part in re.split(r"[\s,]+", sys.argv[3].strip()) if part]
target_col = sys.argv[4]
lookback = int(sys.argv[5])
horizon = int(sys.argv[6])
stride = int(sys.argv[7])

included = []
excluded = []

for domain in candidates:
    try:
        train_records = load_timemmd_windows(
            TimeMMDWindowConfig(
                root_dir=timemmd_root,
                domain=domain,
                lookback=lookback,
                horizon=horizon,
                stride=stride,
                target_col=target_col,
                max_windows=1,
                split="train",
                val_ratio=0.1,
                test_ratio=0.1,
            )
        )
        if not train_records:
            excluded.append({"domain": domain, "reason": "no_train_windows"})
            continue

        test_records = load_timemmd_windows(
            TimeMMDWindowConfig(
                root_dir=timemmd_root,
                domain=domain,
                lookback=lookback,
                horizon=horizon,
                stride=stride,
                target_col=target_col,
                max_windows=1,
                split="test",
                val_ratio=0.1,
                test_ratio=0.1,
            )
        )
        if not test_records:
            excluded.append({"domain": domain, "reason": "no_test_windows"})
            continue
    except Exception as exc:
        excluded.append({"domain": domain, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    included.append(domain)

payload = {
    "experiment_name": "timemmd_experiment_1",
    "target_col": target_col,
    "lookback": lookback,
    "horizon": horizon,
    "stride": stride,
    "candidates": candidates,
    "included_domains": included,
    "excluded_domains": excluded,
}
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

DOMAIN_LIST=()
while IFS= read -r domain_name; do
  if [[ -n "$domain_name" ]]; then
    DOMAIN_LIST+=("$domain_name")
  fi
done < <(python3 - "$DOMAIN_SELECTION_PATH" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for domain in payload.get("included_domains", []):
    print(domain)
PY
)

if [[ "${#DOMAIN_LIST[@]}" -eq 0 ]]; then
  echo "No compatible Time-MMD domains remained after preflight. See $DOMAIN_SELECTION_PATH" >&2
  exit 1
fi

JOINT_DOMAINS="${DOMAIN_LIST[*]}"
EXCLUDED_DOMAIN_TEXT="$(python3 - "$DOMAIN_SELECTION_PATH" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
items = [f"{item['domain']}({item['reason']})" for item in payload.get('excluded_domains', [])]
print(" ".join(items))
PY
)"

PRETRAIN_OUTPUT_DIR="$OUTPUT_ROOT/pretrain_checkpoint"
POST_PRETRAIN_EVAL_OUTPUT_DIR="$OUTPUT_ROOT/post_pretrain_eval"
SFT_OUTPUT_DIR="$OUTPUT_ROOT/sft_checkpoint"
POST_SFT_EVAL_OUTPUT_DIR="$OUTPUT_ROOT/post_sft_eval"

python3 - "$MANIFEST_PATH" "$TIMEMMD_ROOT" "$JOINT_DOMAINS" "$DOMAIN_SELECTION_PATH" <<'PY'
import json
import pathlib
import re
import sys

manifest_path = pathlib.Path(sys.argv[1])
timemmd_root = sys.argv[2]
domains = [part for part in re.split(r"[\s,]+", sys.argv[3].strip()) if part]
domain_selection = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="utf-8"))
payload = {
    "experiment_name": "timemmd_experiment_1",
    "protocol": {
        "description": "Joint cross-domain Time-MMD pretrain on train split, then SFT on train split, with final evaluation on test split.",
        "domains": domains,
        "excluded_domains": domain_selection.get("excluded_domains", []),
        "train_split": "train",
        "pretrain_eval_split": "val",
        "sft_eval_split": "val",
        "final_eval_split": "test",
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    },
    "timemmd_root": timemmd_root,
    "domain_selection": domain_selection,
    "notes": [
        "Experiment 1 fixes the protocol, output structure, and summary generation.",
        "Domains are selected automatically by a preflight check requiring non-empty train and test windows under the current target/lookback/horizon configuration.",
    ],
}
manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

printf 'Running %s with %s domains\n' "$EXPERIMENT_NAME" "${#DOMAIN_LIST[@]}"
printf 'Domains: %s\n' "$JOINT_DOMAINS"
printf 'Excluded domains: %s\n' "$EXCLUDED_DOMAIN_TEXT"
printf 'Output root: %s\n' "$OUTPUT_ROOT"

TIMEMMD_ROOT="$TIMEMMD_ROOT" \
DOMAINS="$JOINT_DOMAINS" \
TARGET_COL="$TARGET_COL" \
LOOKBACK="$LOOKBACK" \
HORIZON="$HORIZON" \
STRIDE="$STRIDE" \
MAX_WINDOWS="$MAX_WINDOWS" \
BATCH_SIZE="$BATCH_SIZE" \
MODEL_NAME="$MODEL_NAME" \
TORCH_DTYPE="$TORCH_DTYPE" \
DEVICE_MAP="$DEVICE_MAP" \
DEVICE="$DEVICE" \
LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY" \
USE_FIXED_SPLITS=1 \
TRAIN_SPLIT=train \
PRETRAIN_EVAL_SPLIT=val \
SFT_EVAL_SPLIT=val \
FINAL_EVAL_SPLIT=test \
SPLIT_VAL_RATIO=0.1 \
SPLIT_TEST_RATIO=0.1 \
PRETRAIN_STEPS="$PRETRAIN_STEPS" \
PRETRAIN_LEARNING_RATE="$PRETRAIN_LEARNING_RATE" \
PRETRAIN_PRED_PROBABILITY="$PRETRAIN_PRED_PROBABILITY" \
PRETRAIN_EVAL_RATIO="$PRETRAIN_EVAL_RATIO" \
PRETRAIN_EVAL_MAX_SAMPLES="$PRETRAIN_EVAL_MAX_SAMPLES" \
POST_PRETRAIN_MAX_SAMPLES="$POST_PRETRAIN_MAX_SAMPLES" \
POST_PRETRAIN_POINT_STRATEGY="$POST_PRETRAIN_POINT_STRATEGY" \
POST_PRETRAIN_POINT_TEMPERATURE="$POST_PRETRAIN_POINT_TEMPERATURE" \
POST_PRETRAIN_POINT_TOP_P="$POST_PRETRAIN_POINT_TOP_P" \
POST_PRETRAIN_NUM_PROB_SAMPLES="$POST_PRETRAIN_NUM_PROB_SAMPLES" \
POST_PRETRAIN_PROB_TEMPERATURE="$POST_PRETRAIN_PROB_TEMPERATURE" \
POST_PRETRAIN_PROB_TOP_P="$POST_PRETRAIN_PROB_TOP_P" \
SFT_STEPS="$SFT_STEPS" \
SFT_LEARNING_RATE="$SFT_LEARNING_RATE" \
SFT_USE_LORA="$SFT_USE_LORA" \
SFT_EVAL_RATIO="$SFT_EVAL_RATIO" \
SFT_EVAL_MAX_SAMPLES="$SFT_EVAL_MAX_SAMPLES" \
POST_SFT_MAX_SAMPLES="$POST_SFT_MAX_SAMPLES" \
POST_SFT_POINT_STRATEGY="$POST_SFT_POINT_STRATEGY" \
POST_SFT_POINT_TEMPERATURE="$POST_SFT_POINT_TEMPERATURE" \
POST_SFT_POINT_TOP_P="$POST_SFT_POINT_TOP_P" \
POST_SFT_NUM_PROB_SAMPLES="$POST_SFT_NUM_PROB_SAMPLES" \
POST_SFT_PROB_TEMPERATURE="$POST_SFT_PROB_TEMPERATURE" \
POST_SFT_PROB_TOP_P="$POST_SFT_PROB_TOP_P" \
BASE_RUN_LABEL="$EXPERIMENT_NAME" \
PRETRAIN_RUN_LABEL="$EXPERIMENT_NAME" \
POST_PRETRAIN_EVAL_RUN_LABEL="$EXPERIMENT_NAME" \
SFT_RUN_LABEL="$EXPERIMENT_NAME" \
POST_SFT_EVAL_RUN_LABEL="$EXPERIMENT_NAME" \
PRETRAIN_OUTPUT_DIR="$PRETRAIN_OUTPUT_DIR" \
POST_PRETRAIN_EVAL_OUTPUT_DIR="$POST_PRETRAIN_EVAL_OUTPUT_DIR" \
SFT_OUTPUT_DIR="$SFT_OUTPUT_DIR" \
POST_SFT_EVAL_OUTPUT_DIR="$POST_SFT_EVAL_OUTPUT_DIR" \
bash "$TEXTTS_DIR/scripts/run_timemmd_full_pipeline.sh"

python3 - "$SUMMARY_JSON" "$SUMMARY_TSV" "$PRETRAIN_OUTPUT_DIR" "$POST_PRETRAIN_EVAL_OUTPUT_DIR" "$SFT_OUTPUT_DIR" "$POST_SFT_EVAL_OUTPUT_DIR" "$JOINT_DOMAINS" "$TIMEMMD_ROOT" "$DOMAIN_SELECTION_PATH" <<'PY'
import json
import pathlib
import re
import sys

summary_json = pathlib.Path(sys.argv[1])
summary_tsv = pathlib.Path(sys.argv[2])
pretrain_dir = pathlib.Path(sys.argv[3])
post_pretrain_eval_dir = pathlib.Path(sys.argv[4])
sft_dir = pathlib.Path(sys.argv[5])
post_sft_eval_dir = pathlib.Path(sys.argv[6])
domains = [part for part in re.split(r"[\s,]+", sys.argv[7].strip()) if part]
timemmd_root = sys.argv[8]
domain_selection = json.loads(pathlib.Path(sys.argv[9]).read_text(encoding="utf-8"))


def read_json(path: pathlib.Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


pretrain_meta = read_json(pretrain_dir / "metadata.json")
post_pretrain_metrics = read_json(post_pretrain_eval_dir / "metrics.json")
sft_meta = read_json(sft_dir / "metadata.json")
post_sft_metrics = read_json(post_sft_eval_dir / "metrics.json")

metric_names = sorted(
    {
        key
        for payload in (post_pretrain_metrics, post_sft_metrics)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }
)
delta = {}
for name in metric_names:
    if name in post_pretrain_metrics and name in post_sft_metrics:
        delta[name] = float(post_sft_metrics[name]) - float(post_pretrain_metrics[name])

summary = {
    "experiment_name": "timemmd_experiment_1",
    "protocol": {
        "description": "Joint cross-domain Time-MMD pretrain on train split, then SFT on train split, with final evaluation on test split.",
        "domains": domains,
        "excluded_domains": domain_selection.get("excluded_domains", []),
        "train_split": "train",
        "pretrain_eval_split": "val",
        "sft_eval_split": "val",
        "final_eval_split": "test",
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    },
    "timemmd_root": timemmd_root,
    "domain_selection": domain_selection,
    "paths": {
        "pretrain_checkpoint_dir": str(pretrain_dir),
        "post_pretrain_eval_dir": str(post_pretrain_eval_dir),
        "sft_checkpoint_dir": str(sft_dir),
        "post_sft_eval_dir": str(post_sft_eval_dir),
    },
    "stages": {
        "pretrain": {
            "metadata": pretrain_meta,
        },
        "post_pretrain_eval": {
            "metrics": post_pretrain_metrics,
        },
        "sft": {
            "metadata": sft_meta,
        },
        "post_sft_eval": {
            "metrics": post_sft_metrics,
        },
    },
    "delta_post_sft_minus_post_pretrain": delta,
}

summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

rows = [
    ("post_pretrain_eval", post_pretrain_metrics, str(post_pretrain_eval_dir), str(pretrain_dir)),
    ("post_sft_eval", post_sft_metrics, str(post_sft_eval_dir), str(sft_dir)),
]
columns = ["stage"] + metric_names + ["eval_dir", "checkpoint_dir"]
with summary_tsv.open("w", encoding="utf-8") as handle:
    handle.write("\t".join(columns) + "\n")
    for stage_name, metrics, eval_dir, checkpoint_dir in rows:
        parts = [stage_name]
        for name in metric_names:
            value = metrics.get(name, "NA")
            parts.append(str(value))
        parts.extend([eval_dir, checkpoint_dir])
        handle.write("\t".join(parts) + "\n")
PY

printf '\nExperiment 1 finished.\n'
printf '  manifest: %s\n' "$MANIFEST_PATH"
printf '  domain selection: %s\n' "$DOMAIN_SELECTION_PATH"
printf '  summary:  %s\n' "$SUMMARY_JSON"
printf '  table:    %s\n' "$SUMMARY_TSV"
