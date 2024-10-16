#!/bin/bash -l
NUM_GPUS=2
unset ROCM_PATH
unset HIP_PATH
unset ROCM_VERSION
CUDA_VISIBLE_DEVICES=0,1

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

train_args_file=$1
echo
# 启动DeepSpeed训练
deepspeed --include localhost:0,1 train.py \
          --train_args_file  $train_args_file

