#!/bin/bash
set -e -x
cd /workspace/avatar/finetrainers
source /workspace/avatar/ftenv/bin/activate
export HF_HOME=/root/hf_local
source /workspace/avatar/.hf_env
export WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# our own quick training/validation configs (49x480x720, single GPU)
cat > /workspace/avatar/cog_training.json <<JSON
{ "datasets":[ { "data_root":"finetrainers/crush-smol","dataset_type":"video","id_token":"PIKA_CRUSH","video_resolution_buckets":[[49,480,720]],"reshape_mode":"bicubic","remove_common_llm_caption_prefixes":true } ] }
JSON
cat > /workspace/avatar/cog_val.json <<JSON
{ "data":[ {"caption":"PIKA_CRUSH A red toy car is crushed by a large hydraulic press, flattening it.","image_path":null,"video_path":null,"num_inference_steps":30,"height":480,"width":720,"num_frames":49,"frame_rate":25} ] }
JSON

torchrun --standalone --nnodes=1 --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" train.py \
  --parallel_backend ptd --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1 \
  --model_name cogvideox --pretrained_model_name_or_path THUDM/CogVideoX-2b \
  --dataset_config /workspace/avatar/cog_training.json --dataset_shuffle_buffer_size 10 \
  --enable_precomputation --precomputation_items 50 --precomputation_once \
  --dataloader_num_workers 0 \
  --flow_weighting_scheme logit_normal \
  --training_type lora --seed 42 --batch_size 1 --train_steps 1000 --rank 64 --lora_alpha 64 \
  --target_modules "(transformer_blocks).*(to_q|to_k|to_v|to_out.0)" \
  --gradient_accumulation_steps 1 --gradient_checkpointing \
  --checkpointing_steps 200 --checkpointing_limit 5 \
  --enable_slicing --enable_tiling \
  --optimizer adamw --lr 1e-4 --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
  --lr_num_cycles 1 --beta1 0.9 --beta2 0.99 --weight_decay 1e-4 --epsilon 1e-8 --max_grad_norm 1.0 \
  --validation_dataset_file /workspace/avatar/cog_val.json --validation_steps 250 \
  --tracker_name ft-cog --output_dir /workspace/avatar/cog_lora_long --init_timeout 600 --nccl_timeout 600 \
  --report_to none
echo "TRAIN_DONE"
