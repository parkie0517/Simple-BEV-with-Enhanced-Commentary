#!/bin/bash
# original file is `eval_lyft.sh`
# this file is modified by me

DATA_DIR="/mnt/ssd2/heejun/dataset/lyft"

MODEL_NAME="16x3_5e-4s_rgb06_01:34:39"

EXP_NAME="eval" # eval 16x3_5e-4s_rgb06_01:34:39 lyft
EXP_NAME="eval" # 416, B16
EXP_NAME="eval" # 1920,1080 base resolution
EXP_NAME="eval" # move centroid down 1 < 24.03662218212872
EXP_NAME="eval" # 1024 instead of 1080 (?)
EXP_NAME="eval"

python eval_lyft.py \
       --exp_name=${EXP_NAME} \
       --batch_size=2 \
       --dset='lyft' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_lyft' \
       --init_dir="checkpoints/${MODEL_NAME}" \
       --res_scale=2 \
       --device_ids=[3]

:<<"End"

Explanation of the flags
       dset: make sure is `lyft`

End