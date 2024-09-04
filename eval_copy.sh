#!/bin/bash
# this file is created by me

DATA_DIR="/mnt/ssd2/heejun/dataset/"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

MODEL_NAME="8x5_5e-4_rgb12_22:43:46"

EXP_NAME="12" # evaluate rgb00 model

python eval_nuscenes.py \
       --batch_size=16 \
       --exp_name=${EXP_NAME} \
       --dset='nuscenes' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_nuscenes' \
       --init_dir="checkpoints/${MODEL_NAME}" \
       --res_scale=2 \
       --device_ids=[3] # use the 4th gpu
