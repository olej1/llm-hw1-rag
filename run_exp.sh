#!/bin/bash
# ./run_exp.sh <name> [questions] [gold] — прогон конфига configs/<name>.json и запись в experiments.csv
set -e
name=$1; q=${2:-data/dev_questions.jsonl}; g=${3:-data/dev_gold.jsonl}
export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8011/v1} LLM_MODEL=${LLM_MODEL:-Qwen3.5-4B} SUBMISSION_ID=${SUBMISSION_ID:-dev-$name}
mkdir -p runs
./.venv/bin/python predict.py --index ./index --questions $q --output runs/$name.answers.jsonl \
  --trace runs/$name.trace.jsonl --config configs/$name.json
extra=""; [ "$name" = no_context ] && extra="--no-q"
./.venv/bin/python evaluate.py --questions $q --gold $g --answers runs/$name.answers.jsonl \
  --trace runs/$name.trace.jsonl --name $name --csv experiments.csv $extra ${CREF:+--cref $CREF}
