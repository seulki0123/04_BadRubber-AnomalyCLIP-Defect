#!/bin/bash

source venv_py38/bin/activate

SRC_ROOT="./NAS_Site_SSBR"
DST_ROOT="./NAS/_report"

LINE="BR-A"
GRADE="1208_for-crops"

DATES=(
  "2026-01-10"
  "2026-02-10"
)

BATCH_SIZE=9

python src/report.py \
  --src-root "$SRC_ROOT" \
  --dst-root "$DST_ROOT" \
  --line "$LINE" \
  --grade "$GRADE" \
  --dates "${DATES[@]}" \
  --batch-size "$BATCH_SIZE"