#!/bin/bash
#SBATCH --job-name=vel_energy
#SBATCH --time=36:00:00
#SBATCH --partition=mi3008xl
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=92

cd /work1/chunyilee/yuhang/openvla-oft-yhs
export WANDB_API_KEY=0bdbd99b1136358467ed2d03e9a6ba5a5b2a11a8
export HF_HOME=/work1/aiginternal/yuhang/
export OMP_NUM_THREADS=11


PRETRAINED_CKPT=openvla/openvla-7b

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_Energy_freeze.py \
  --vla_path              $PRETRAINED_CKPT \
  --data_root_dir         /work1/chunyilee/yuhang/modified_libero_rlds \
  --dataset_name          libero_4_task_suites_no_noops \
  --run_root_dir          /work1/chunyilee/yuhang/openvla-energy/ckpoints \
  --use_l1_regression     True \
  --use_diffusion         False \
  --use_film              False \
  --num_images_in_input   2 \
  --use_proprio           True \
  --batch_size            8 \
  --energy_learning_rate  5e-4 \
  --energy_warm_steps     0 \
  --num_steps_before_decay 100000 \
  --max_steps             50005 \
  --save_freq             5000 \
  --save_latest_checkpoint_only False \
  --image_aug             True \
  --lora_rank             32 \
  --wandb_entity          "yhscode-university-of-liverpool" \
  --wandb_project         "energyvla" \
  --run_id_note           vel_gal_multiscale
