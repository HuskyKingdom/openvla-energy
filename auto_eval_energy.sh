#!/bin/bash

# VEL v2 (P1) · eval run: vel_gal_multiscale / 50000 step
# - Base VLA & action head & proprio projector: HF hub (training was freeze-mode,
#   so these are identical to the upstream moojink checkpoint).
# - Energy head: loaded from our training run_root_dir via --energy_ckpt.
# - energy_alpha=0.2 = α_max; line-search grid becomes (0.2, 0.1, 0.05, 0.0).

echo "Running Evaluations Automatically ------------------------------"

PRETRAINED_CKPT=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
ENERGY_CKPT=ckpoints/energy_model--50000_checkpoint.pt
ENERGY_ALPHA=0.2
RUN_TAG=velv2_p1_50k

echo "Evaluating spatial ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint $PRETRAINED_CKPT \
    --energy_ckpt           $ENERGY_CKPT \
    --task_suite_name       libero_spatial \
    --e_decoding            True \
    --energy_alpha          $ENERGY_ALPHA \
    --task_label            ${RUN_TAG}_spatial

echo "Evaluating object ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint $PRETRAINED_CKPT \
    --energy_ckpt           $ENERGY_CKPT \
    --task_suite_name       libero_object \
    --e_decoding            True \
    --energy_alpha          $ENERGY_ALPHA \
    --task_label            ${RUN_TAG}_object

echo "Evaluating goal ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint $PRETRAINED_CKPT \
    --energy_ckpt           $ENERGY_CKPT \
    --task_suite_name       libero_goal \
    --e_decoding            True \
    --energy_alpha          $ENERGY_ALPHA \
    --task_label            ${RUN_TAG}_goal

echo "Evaluating long ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint $PRETRAINED_CKPT \
    --energy_ckpt           $ENERGY_CKPT \
    --task_suite_name       libero_10 \
    --e_decoding            True \
    --energy_alpha          $ENERGY_ALPHA \
    --task_label            ${RUN_TAG}_long
