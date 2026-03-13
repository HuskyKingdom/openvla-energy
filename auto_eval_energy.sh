#!/bin/bash

# without energy
echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10


echo "Evaluating spatial ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained \
    --task_suite_name libero_spatial --e_decoding True --task_label w_energy_spatial_0.3

echo "Evaluating object ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained \
    --task_suite_name libero_object --e_decoding True --task_label w_energy_object_0.3

echo "Evaluating goal ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained \
    --task_suite_name libero_goal --e_decoding True --task_label w_energy_goal_0.3

echo "Evaluating long ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained \
    --task_suite_name libero_10 --e_decoding True --task_label w_energy_10_0.3


