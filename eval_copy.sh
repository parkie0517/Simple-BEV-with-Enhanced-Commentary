#!/bin/bash
# this file is created by me

DATA_DIR="/mnt/ssd2/heejun/dataset/nuScenes"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

MODEL_NAME="8x5_5e-4_rgb12_22:43:46"

EXP_NAME="00" # evaluate rgb00 model

python eval_nuscenes.py \
       --batch_size=16 \
       --exp_name=${EXP_NAME} \
       --dset='mini' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_nuscenes' \
       --init_dir="checkpoints/${MODEL_NAME}" \
       --res_scale=1 \
       --device_ids=[0]
