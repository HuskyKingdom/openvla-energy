#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --time=02:00:00
#SBATCH --partition=vlm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 92

cd /work1/aiginternal/yuhang/openvla-oft-yhs
export WANDB_API_KEY=0bdbd99b1136358467ed2d03e9a6ba5a5b2a11a8
export HF_HOME=/work1/aiginternal/yuhang/



FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_withHNN.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /work1/aiginternal/yuhang/modified_libero_rlds \
  --dataset_name libero_4_task_suites_no_noops \
  --run_root_dir /work1/aiginternal/yuhang/openvla-oft-yhs/ckpoints \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 26 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 250005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "yhscode-university-of-liverpool" \
  --wandb_project "yhscode-university-of-liverpool" \
  --run_id_note hnn0.3_uni_Psteps