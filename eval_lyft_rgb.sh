#!/bin/bash
# original file is `eval_lyft.sh`
# this file is modified by me

DATA_DIR="/mnt/ssd2/heejun/dataset/lyft"


MODEL_NAME="8x5_5e-4_rgb12_22:43:46"


EXP_NAME="eval"


:<<"End1"

# eval 16x3_5e-4s_rgb06_01:34:39 lyft
# 416, B16
# 1920,1080 base resolution
# move centroid down 1 < 24.03662218212872
# 1024 instead of 1080 (?)

End1


python eval_lyft.py \
       --exp_name=${EXP_NAME} \
       --batch_size=1 \
       --dset='lyft' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_lyft' \
       --init_dir="checkpoints/${MODEL_NAME}" \
       --res_scale=1 \
       --device_ids=[3] \
       --mini=False


:<<"End2"

Explanation of the flags
       dset: make sure is `lyft`
       mini: if set to `True` then, only evaulate a portion of the validation dataset

End2