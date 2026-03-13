#!/bin/bash

# without energy
echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10


echo "Evaluating spatial ------------------------------"
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained  \
    --task_suite_name libero_spatial --e_decoding True  --energy_alpha 1.0 --task_label energy_alpha_1.0_spatial


python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained  \
    --task_suite_name libero_spatial --e_decoding True  --energy_alpha 0.5 --task_label energy_alpha_0.5_spatial


echo "Evaluating long ------------------------------"
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained  \
    --task_suite_name libero_10 --e_decoding True  --energy_alpha 1.0 --task_label energy_alpha_1.0_10


python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained  \
    --task_suite_name libero_10 --e_decoding True  --energy_alpha 0.5 --task_label energy_alpha_0.5_10



echo "Evaluating goal ------------------------------"
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained  \
    --task_suite_name libero_goal --e_decoding True  --energy_alpha 1.0 --task_label energy_alpha_1.0_goal


python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained  \
    --task_suite_name libero_goal --e_decoding True  --energy_alpha 0.5 --task_label energy_alpha_0.5_goal



echo "Evaluating object ------------------------------"
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained  \
    --task_suite_name libero_object --e_decoding True --energy_alpha 1.0 --task_label energy_alpha_1.0_object


python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint /home/aup/YuhangWorkspace/openvla-oft-yhs/ckpts/pre-trained  \
    --task_suite_name libero_object --e_decoding True --energy_alpha 0.5 --task_label energy_alpha_0.5_object


