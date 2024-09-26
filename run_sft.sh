export CUDA_VISIBLE_DEVICES=3,4

torchrun --nnodes 1 --nproc_per_node 2 --master_port 25641 \
/mapping-data/qianli/firefly/train.py \
--train_args_file /mapping-data/qianli/firefly/train_args/sft/full/qwen2-1.5b-sft-full.json
