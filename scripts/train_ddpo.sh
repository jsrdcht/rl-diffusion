export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE="offline"


CUDA_VISIBLE_DEVICES=1,5 accelerate launch --multi_gpu --num_processes=2 --main_process_port 29503 scripts/train.py