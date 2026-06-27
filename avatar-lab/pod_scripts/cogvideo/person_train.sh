#!/bin/bash
set -x
cd /workspace/avatar/finetrainers
source /workspace/avatar/ftenv/bin/activate
export HF_HOME=/root/hf_local
source /workspace/avatar/.hf_env
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=offline
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nnodes=1 --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" train.py \
  --parallel_backend ptd --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1 \
  --model_name cogvideox --pretrained_model_name_or_path THUDM/CogVideoX-2b \
  --dataset_config /workspace/avatar/person_training.json --dataset_shuffle_buffer_size 10 \
  --enable_precomputation --precomputation_items 20 --precomputation_once \
  --dataloader_num_workers 0 --flow_weighting_scheme logit_normal \
  --training_type lora --seed 42 --batch_size 1 --train_steps 1500 --rank 64 --lora_alpha 64 \
  --target_modules "(transformer_blocks).*(to_q|to_k|to_v|to_out.0)" \
  --gradient_accumulation_steps 1 --gradient_checkpointing \
  --checkpointing_steps 250 --checkpointing_limit 6 \
  --enable_slicing --enable_tiling \
  --optimizer adamw --lr 1e-4 --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
  --lr_num_cycles 1 --beta1 0.9 --beta2 0.99 --weight_decay 1e-4 --epsilon 1e-8 --max_grad_norm 1.0 \
  --validation_dataset_file /workspace/avatar/person_val.json --validation_steps 250 \
  --tracker_name ft-person --output_dir /workspace/avatar/person_lora_out --init_timeout 600 --nccl_timeout 600 \
  --report_to none
echo PERSON_TRAIN_DONE
