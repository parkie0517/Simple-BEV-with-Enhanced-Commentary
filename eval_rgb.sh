#!/bin/bash
# this file is created by me

DATA_DIR="/mnt/ssd2/heejun/dataset/nuscenes"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

MODEL_NAME="8x5_5e-4_rgb12_22:43:46"

EXP_NAME="12" # evaluate rgb00 model

python eval_nuscenes.py \
       --batch_size=4 \
       --exp_name=${EXP_NAME} \
       --dset='trainval' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_nuscenes' \
       --init_dir="checkpoints/${MODEL_NAME}" \
       --use_radar=False \
       --res_scale=2 \
       --device_ids=[3]

:<<"End"

Explanation of the flags
dset: should be either one of `mini` or `trainval`
res_scale: used to increase the size of the rgb resolution
       ex) input rgb size: (224, 400) x res_scale
device_ids: gpus that you wish to use
       ex) `[3]` means that you only want to use the 4th gpu. `[0, 1, 2, 3]' means that you want to use gpu from 1 to 4. 

End