#!/bin/bash

# without energy
echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10


echo "Evaluating ------------------------------"
python experiments/robot/libero/run_libero_eval_cus.py \
    --pretrained_checkpoint ckpts/openvla-7b+libero_4_task_suites_no_noops+b3+lr-0.0005+lora-r32+dropout-0.0--image_aug--rephrasevla--200000_chkpt/ \
     --task_suite_name libero_10  --e_decoding False --save_video False --task_label empty_false --cus_task empty --remove_wrap False

python experiments/robot/libero/run_libero_eval_cus.py \
    --pretrained_checkpoint ckpts/openvla-7b+libero_4_task_suites_no_noops+b3+lr-0.0005+lora-r32+dropout-0.0--image_aug--rephrasevla--200000_chkpt/ \
     --task_suite_name libero_10  --e_decoding False --save_video False --task_label empty_true --cus_task empty --remove_wrap True

python experiments/robot/libero/run_libero_eval_cus.py \
    --pretrained_checkpoint ckpts/openvla-7b+libero_4_task_suites_no_noops+b3+lr-0.0005+lora-r32+dropout-0.0--image_aug--rephrasevla--200000_chkpt/ \
     --task_suite_name libero_10  --e_decoding False --save_video False --task_label none_true --remove_wrap True


python experiments/robot/libero/run_libero_eval_cus.py \
    --pretrained_checkpoint ckpts/openvla-7b+libero_4_task_suites_no_noops+b3+lr-0.0005+lora-r32+dropout-0.0--image_aug--rephrasevla--200000_chkpt/ \
     --task_suite_name libero_spatial  --e_decoding False --save_video False --task_label empty_false_s --cus_task empty --remove_wrap False

python experiments/robot/libero/run_libero_eval_cus.py \
    --pretrained_checkpoint ckpts/openvla-7b+libero_4_task_suites_no_noops+b3+lr-0.0005+lora-r32+dropout-0.0--image_aug--rephrasevla--200000_chkpt/ \
     --task_suite_name libero_spatial  --e_decoding False --save_video False --task_label empty_true_s --cus_task empty --remove_wrap True

python experiments/robot/libero/run_libero_eval_cus.py \
    --pretrained_checkpoint ckpts/openvla-7b+libero_4_task_suites_no_noops+b3+lr-0.0005+lora-r32+dropout-0.0--image_aug--rephrasevla--200000_chkpt/ \
     --task_suite_name libero_spatial  --e_decoding False --save_video False --task_label none_true_s --remove_wrap True


