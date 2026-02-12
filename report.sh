#!/bin/bash

source venv_py38/bin/activate

# SRC_ROOT="./NAS"
# DST_ROOT="./NAS_Site_SSBR"
SRC_ROOT="./tests/datasets"
DST_ROOT="./tests/_report"

LINE="SSBR"
GRADE="FFFFF"

DATES=(
  "2025-11-21"
  # "2025-11-22"
)

BATCH_SIZE=9

python src/report.py \
  --src-root "$SRC_ROOT" \
  --dst-root "$DST_ROOT" \
  --line "$LINE" \
  --grade "$GRADE" \
  --dates "${DATES[@]}" \
  --batch-size "$BATCH_SIZE"