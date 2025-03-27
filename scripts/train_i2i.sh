export HF_ENDPOINT=https://hf-mirror.com
export NCCL_IB_TIMEOUT=20  # 设置为60秒
export NCCL_DEBUG=INFO
export WANDB_MODE="offline"
conda activate ddpo_env

CUDA_VISIBLE_DEVICES=6,7 accelerate launch --multi_gpu --num_processes=2 scripts/train_i2i.py