#!/bin/bash -l

CUDA_VISIBLE_DEVICES=1 nohup python start_service.py 2>&1 > base.log &
CUDA_VISIBLE_DEVICES=1 nohup python start_service.py --model_name_or_path /mapping-data/qianli/firefly/Qwen2-1.5B --port 2986 --log_file service_sft_history.log 2>&1 > sft.log &
