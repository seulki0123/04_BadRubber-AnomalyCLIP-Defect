#!/bin/bash

# ================== 설정 ==================
DEVICE=0

DEPTH=9
N_CTX=12
T_N_CTX=4

DATASET=mvtec
DATA_PATH=./datasets/dataset_final_for_anomaly

BASE_DIR=${DEPTH}_${N_CTX}_${T_N_CTX}_multiscale_mvtec_260204_mvtec_anomaly_detection+F1038-F2150-M2520
SAVE_DIR=./checkpoints/${BASE_DIR}

# ================== 실행 ==================
echo "Save dir: ${SAVE_DIR}"

CUDA_VISIBLE_DEVICES=${DEVICE} python src/AnomalyCLIP/train.py \
  --dataset ${DATASET} \
  --train_data_path ${DATA_PATH} \
  --save_path ${SAVE_DIR} \
  --features_list 24 \
  --image_size 518 \
  --batch_size 8 \
  --print_freq 1 \
  --epoch 15 \
  --save_freq 1 \
  --depth ${DEPTH} \
  --n_ctx ${N_CTX} \
  --t_n_ctx ${T_N_CTX}
