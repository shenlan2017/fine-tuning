#!/bin/bash -l

# 设置路径和参数

NUM_GPUS=2
unset ROCM_PATH
unset HIP_PATH
unset ROCM_VERSION
CUDA_VISIBLE_DEVICES=0,1
train_args_file=$1

# 启动DeepSpeed训练
deepspeed --include localhost:0,1 train_tools.py \
        --train_args_file $train_args_file \
