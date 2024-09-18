# 环境安装
## Docker
- docker 安装：curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
- docker 启动：service docker restart
- cd docker/docker-cuda/
- docker compose up -d
- docker compose exec llamafactory bash
- pip install -r requirements.txt

# 摘要生成

## *数据准备*
- 训练集：`firefly/data/summary/sum_train_2048.jsonl`
- 测试集：`firefly/data/summary/summary_test.jsonl`
- 模型：`Qwen2-1.5B`

## *Train*
```
bash train.sh train_args/sft/lora/qwen1.5-7b-sft-lora.json
```

其中 `train_args/sft/lora/qwen1.5-7b-sft-lora.json` 为训练参数文件，可自行修改：
- LoRA 训练文件为：`train_args/sft/lora/qwen1.5-7b-sft-lora.json`
- Full-finetune 训练文件为：`train_args/sft/full/qwen2-1.5b-sft-full.json`
- 训练参数文件：
  - `model_name_or_path: 输入模型地址`,
  - `deepspeed: deepspeed配置文件地址：train_args/ds_z3_config.json`,
  - `output_dir: 模型保存地址 `,
  - `template_name: 对话模板qwen`,
  - `train_mode: full/lora`,
  -  `num_train_epochs: 1`,
  -  `per_device_train_batch_size: 1`,
   - `gradient_accumulation_steps: 2,梯度累积次数`,

## *模型文件和生成结果文件:*

### *Full finetune：*
 - CKPT: `firefly/output_summary_demo/qwen2-1.5b-sft-full`
 - 生成结果：`firefly/output_summary_demo/qwen2-1.5b-sft-full/full/summary_test-res.jsonl` 
### *LoRA：*
 - CKPT: `firefly/output_summary_demo/firefly-qwen2-1.5b-sft-lora`
 - 生成结果：`firefly/output_summary_demo/firefly-qwen2-1.5b-sft-lora/lora/summary_test-res.jsonl`   



## *Inference*
### *批量推理模式*
```
cd scripts/chat  
bash ./Summary_infer.sh  path/to/model_path path/to/input_file path/to/output_dir
或者
python batch_generate.py --model_path path/to/model_path --input_file path/to/input_file --output_dir path/to/output_dir
```

其中：
- --model_path 模型路径  
- --input_file 测试文件路径  
- --output_dir 输出文件路径

### *交互式问答模式*

```
cd scripts/chat  
bash ./chat.sh path/to/model_path 
或者
python chat.py --model_path path/to/model_path
```
### *指标计算*
摘要生成的指标为BLEU-1/2/3/4，ROUGE-1/2-L

```
cd script/evaluate

bash summary_score.sh path/to/ground_truth_file path/to/generation_file

或者

python score.py --ground_truth_file firefly/data/summary/summary_test.jsonl --generated_file firefly/output_summary_demo/qwen2-1.5b-sft-full/full/summary_test-res.jsonl
```



# Tools call（工具调用）


## *数据准备*
- 训练集：`firefly/data/tools/glaive_toolcall_zh_1k_train.jsonl`
- 测试集：`firefly/data/tools/glaive_toolcall_zh_1k_test.jsonl`
- tools类型文件，训练时随机抽取不同的10个tools其中包括正确的tool: `firefly/data/tools/extracted_query_reply.jsonl`
- 模型：`Qwen2-1.5B`


## *Train*

```
bash train_tools.sh train_args/sft/lora/qwen1.5-7b-sft-lora-tools.json
```

其中 `train_args/sft/lora/qwen1.5-7b-sft-lora-tools.json` 为训练参数文件，可自行修改：
- LoRA 训练文件为：`train_args/sft/lora/qwen1.5-7b-sft-lora-tools.json`
- Full-finetune 训练文件为：`train_args/sft/full/qwen2-1.5b-sft-full-tools.json`
- 训练参数文件：
  - `model_name_or_path: 输入模型地址`,
  - `deepspeed: deepspeed配置文件地址：train_args/ds_z3_config.json`,
  - `output_dir: 模型保存地址 `,
  - `template_name: qwen-tools`,
  - `train_mode: full/lora`,
  -  `num_train_epochs: 1`,
  -  `per_device_train_batch_size: 1`,
   - `gradient_accumulation_steps: 2,梯度累积次数`,



## *模型文件和生成结果文件:*
### *Full finetune：*
 - CKPT: `firefly/output_tools_call_demo/qwen2-1.5b-sft-full-tools-1k-train-1800`
 - 生成结果文件：`firefly/output_tools_call_demo/qwen2-1.5b-sft-full-tools-1k-train-1800/full/glaive_toolcall_zh_1k_test-res.jsonl` 
### *LoRA：*
 - CKPT: `firefly/output_tools_call_demo/firefly-qwen2-1.5b-sft-lora-tools-1800`
 - 生成结果文件： `firefly/output_tools_call_demo/firefly-qwen2-1.5b-sft-lora-tools-1800/lora/glaive_toolcall_zh_1k_test-res.jsonl`   

## *推理*

### *批量推理模式*

```
cd scripts/chat  
bash ./Tools_infer.sh  path/to/model_path path/to/input_file path/to/output_dir
或者
python batch_generate_tools.py --model_path path/to/model_path --input_file path/to/input_file --output_dir path/to/output_dir
```

其中：
- --model_path 模型路径  
- --input_file 测试文件路径  
- --output_dir 输出文件路径

### *交互式问答模式*

```
cd scripts/chat 
bash ./chat_tools.sh path/to/model_path 
或者
python tool_chat.py --model_path path/to/model_path
```

### *指标计算*
工具调用的指标为调用的工具是否准确，因此采用准确率作为计算方法：
```
cd script/evaluate
python tools_score.py --input_file firefly/output_tools_call_demo/qwen2-1.5b-sft-full-tools-1k-train-1800/full/glaive_toolcall_zh_1k_test-res.jsonl
或
bash ./tools_score.sh  firefly/output_tools_call_demo/qwen2-1.5b-sft-full-tools-1k-train-1800/full/glaive_toolcall_zh_1k_test-res.jsonl 
```
