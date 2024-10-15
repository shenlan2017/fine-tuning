# 环境安装
## Docker
- docker 安装：curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
- docker 启动：service docker restart
- cd docker/docker-cuda/
- docker compose up -d
- docker compose exec llamafactory bash
- pip install -r requirements.txt
- pip install --no-cache-dir flash-attn==2.1.1 --no-build-isolation

# 摘要生成

## *数据准备*
- 训练集：`fine-tuning/data/summary/sum_train_2048.jsonl`
- 测试集：`fine-tuning/data/summary/summary_test.jsonl`
- 模型：`Qwen2-1.5B`

## *Train*
```
bash train.sh train_args/sft/lora/qwen2-1.5b-sft-lora.json
```

其中 `train_args/sft/lora/qwen2-1.5b-sft-lora.json` 为训练参数文件，可自行修改：
- LoRA 训练文件为：`train_args/sft/lora/qwen2-1.5b-sft-lora.json`
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
 - CKPT: `fine-tuning/output_summary_demo/qwen2-1.5b-sft-full`
 - 生成结果：`fine-tuning/output_summary_demo/qwen2-1.5b-sft-full/full/summary_test-res.jsonl` 
### *LoRA：*
 - CKPT: `fine-tuning/output_summary_demo/qwen2-1.5b-sft-lora`
 - 生成结果：`fine-tuning/output_summary_demo/qwen2-1.5b-sft-lora/lora/summary_test-res.jsonl`   



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

python score.py --ground_truth_file fine-tuning/data/summary/summary_test.jsonl --generated_file fine-tuning/output_summary_demo/qwen2-1.5b-sft-full/full/summary_test-res.jsonl
```



# Tools call（工具调用）


## *数据准备*
- 训练集：`fine-tuning/data/tools/glaive_toolcall_zh_1k_train.jsonl`
- 测试集：`fine-tuning/data/tools/glaive_toolcall_zh_1k_test.jsonl`
- tools类型文件，训练时随机抽取不同的10个tools其中包括正确的tool: `fine-tuning/data/tools/extracted_query_reply.jsonl`
- 模型：`Qwen2-1.5B`


## *Train*

```
bash train_tools.sh train_args/sft/lora/qwen2-1.5b-sft-lora-tools.json
```

其中 `train_args/sft/lora/qwen2-1.5b-sft-lora-tools.json` 为训练参数文件，可自行修改：
- LoRA 训练文件为：`train_args/sft/lora/qwen2-1.5b-sft-lora-tools.json`
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
 - CKPT: `fine-tuning/output_tools_call_demo/qwen2-1.5b-sft-full-tools-1k-train-1800`
 - 生成结果文件：`fine-tuning/output_tools_call_demo/qwen2-1.5b-sft-full-tools-1k-train-1800/full/glaive_toolcall_zh_1k_test-res.jsonl` 
### *LoRA：*
 - CKPT: `fine-tuning/output_tools_call_demo/qwen2-1.5b-sft-lora-tools-1800`
 - 生成结果文件： `fine-tuning/output_tools_call_demo/qwen2-1.5b-sft-lora-tools-1800/lora/glaive_toolcall_zh_1k_test-res.jsonl`   

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
python tools_score.py --input_file fine-tuning/output_tools_call_demo/qwen2-1.5b-sft-full-tools-1k-train-1800/full/glaive_toolcall_zh_1k_test-res.jsonl
或
bash ./tools_score.sh  fine-tuning/output_tools_call_demo/qwen2-1.5b-sft-full-tools-1k-train-1800/full/glaive_toolcall_zh_1k_test-res.jsonl 
```
# 使用GPT进行数据生成
```
cd scripts/data_generation 
```
使用前需要更换为自己的api: gpt4_summary.py文件的第16行
```
client = OpenAI(
    api_key="",  # 替换成真实DashScope的API_KEY
    base_url="",  # 填写DashScope服务endpoint
)
```
```
python gpt4_summary.py --input_file path/to/input_file --output_file path/to/output_file
```
其中：
- --input_file 输入文件路径，格式可以为jsonl,csv等，src字段作为请求的query。  
- --output_dir 输出文件路径

以摘要数据构建为例，GPT请求和返回分别如下所示:

构造摘要数据输入示例(src)：
```
{"src": "提炼以下文本的主旨并注意保留原文中的原始标号：3.4.2 企业双碳咨询认证服务\n企业双碳咨询，是指企业根据其行业特点及发展阶段制定对应的碳中和规划，更好地平衡企业碳减排和发展之间的关系。从碳盘查及认证的维度来看，可分为组织碳足迹及产品碳足迹。目前 就 国 际 贸 易、ESG 披 露 及 绿 色 金 融 等 市 场 来 看， 企 业 对 碳 足 迹 认 证 的 需 求 也 越 来 越 强 烈， 对碳足迹数字化平台的需求也越来越迫切。\n企业碳中和规划 : 企业通过数字化平台提交相关资料，平台模拟分析，摸清企业碳排放家底、分析企业碳排放现状问题和减排潜力。把握政府和市场、长期和短期、整体和局部、发展和减排的关系，预测企业未来发展及碳排放情况；结合当前社会低碳发展的趋势和政策，分析碳中和的目标；依据规划碳中和的目标，综合考虑机制体制、能源结构、能源效率、工艺设备和低碳技术等方面的情景，模拟计算不同情形的碳排放，根据情况制订碳中和的路径；规划实施碳中和的重大项目，方便企业实施碳中和项目，实现碳中和目标。\n企业碳足迹：为企业提供碳足迹的在线数字化认证。通过企业在线提供碳排信息，对企业温室气体排放进行核算分析，寻找节约能源、减少碳排放与技术改造的机会，提高能源利用效率，减少温室气体排放，应对政策壁垒，规避法律风险。通过向利益相关方的公开报告，展示组织的低碳产品及服务，提高利益相关方的认可程度，提升企业社会形象。\n产品碳足迹：帮助企业进行产品碳足迹在线认证，降低企业整个低碳核算和认证的成本，同时 利 用 区 块 链 和 IOT 技 术， 对 原 材 料 开 采、 产 品 生 产（ 或 服 务 提 供）、 运 输、  用 到 最 终 再 生利用和处置等多个阶段进行系统化、定量化评价产品生命周期过程中资源环境效率，实现整个数据和结果的可信可追溯。企业基于产品碳足迹的全面评价，为后续实现减排目标和持续减排方案提供了基础；在外贸场景下，能够为企业避免碳关税带来的出口困难；碳足迹结果可用于下游产品设计与供应链低碳管理，促进全产业链的低碳发展；碳足迹结果可用于市场宣传，展示产品或生产工艺在碳排放方面的优势，为下游厂商或终端消费者的低碳选择提供依据。"}
```
GPT生成内容(tgt)：
```
3.4.2 企业双碳咨询认证服务\n\n企业双碳咨询，旨在帮助企业根据其行业特点及发展阶段制定对应的碳中和规划，实现碳减排与发展的平衡。碳盘查及认证可以分为组织碳足迹及产品碳足迹两大类，目前，企业对碳足迹认证的需求随国际贸易、ESG披露及绿色金融等市场需求增强而增加。\n\n企业碳中和规划：企业通过数字化平台提交资料，进行碳排放模拟分析，理解碳排放状况及减排潜力，预测未来发展及碳排放情况。规划中将考虑多个方面，如机制体制、能源结构等，模拟不同情形的碳排放，制订碳中和路径，规划实施碳中和重大项目。\n\n企业碳足迹：提供在线数字化碳足迹认证，通过企业提供的碳排信息，分析温室气体排放，寻找减排机会，提高能源利用效率，应对政策壁垒，规避法律风险，同时提升企业社会形象。\n\n产品碳足迹：通过在线认证降低低碳核算和认证成本，利用区块链和IOT技术，对产品全生命周期进行系统化、定量化评价，实现数据和结果的可信可追溯。评价结果可以帮助企业实现减排目标，避免碳关税带来的出口困难，促进全产业链的低碳发展，同时为市场宣传提供依据。
```
