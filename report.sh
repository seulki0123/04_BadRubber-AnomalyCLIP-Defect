#!/bin/bash

source venv_py310/bin/activate

SRC_ROOT="/home/s1k2/04_BadRubber/datasets/raw"
DST_ROOT="/home/s1k2/04_BadRubber/datasets/processed"

LINE="BR-A"
GRADE="unknown"

DATES=(
  "2026-02-18"
)

BATCH_SIZE=9

python src/report.py \
  --src-root "$SRC_ROOT" \
  --dst-root "$DST_ROOT" \
  --line "$LINE" \
  --grade "$GRADE" \
  --dates "${DATES[@]}" \
  --batch-size "$BATCH_SIZE"