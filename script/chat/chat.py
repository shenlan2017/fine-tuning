import argparse

from transformers import AutoTokenizer, AutoConfig, AddedToken
import torch
from loguru import logger
import copy

import sys
sys.path.append("../../")
from component.utils import ModelUtils
from component.template import template_dict

def get_inference_args():
    args = argparse.ArgumentParser("Inference Arguments")
    args.add_argument("--model_path", type=str, default="")
    return args.parse_args()
def build_prompt_chatglm3(tokenizer, query, history, system=None):
    history.append({"role": 'user', 'message': query})
    # system
    input_ids = tokenizer.get_prefix_tokens() + \
                [tokenizer.get_command(f"<|system|>")] + \
                tokenizer.encode(system, add_special_tokens=False)
    # convs
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            tokens = [tokenizer.get_command(f"<|user|>")] + \
                     tokenizer.encode(message, add_special_tokens=False) + \
                     [tokenizer.get_command(f"<|assistant|>")]
        else:
            tokens = tokenizer.encode(message, add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids += tokens

    return input_ids


def build_prompt(tokenizer, template, query, history, system=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    if template_name == 'chatglm2':
        prompt = tokenizer.build_prompt(query, history)
        input_ids = tokenizer.encode(prompt)
    elif template_name == 'chatglm3':
        input_ids = build_prompt_chatglm3(tokenizer, query, history, system)
    else:
        history.append({"role": 'user', 'message': query})
        input_ids = []

        # setting system information
        if system_format is not None:
            # system信息不为空
            if system is not None:
                system_text = system_format.format(content=system)
                input_ids = tokenizer.encode(system_text, add_special_tokens=False)
        # concat conversation
        for item in history:
            role, message = item['role'], item['message']
            if role == 'user':
                message = user_format.format(content=message, stop_token=tokenizer.eos_token)
            else:
                message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
            tokens = tokenizer.encode(message, add_special_tokens=False)
            input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


def load_tokenizer(model_name_or_path):
    # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
        # llama不支持fast
        # use_fast=False if config.model_type == 'llama' else True
    )

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    return tokenizer


def main():
    # 使用合并后的模型进行推理
    # model_name_or_path = 'Qwen/Qwen-7B-Chat'
    # template_name = 'qwen'
    #  adapter_name_or_path = None
    args = get_inference_args()
    # model_name_or_path = '/mapping-data/qianli/firefly/Qwen2-1.5B'  # Qwen1.5-1.8B Qwen2-1.5B
    model_name_or_path =args.model_path

    template_name = 'qwen'
    adapter_name_or_path = None

    template = template_dict[template_name]
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0

    # 加载模型
    logger.info(f'Loading model from: {model_name_or_path}')
    logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
    if template_name == 'chatglm2':
        stop_token_id = tokenizer.eos_token_id
    elif template_name == 'chatglm3':
        stop_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]
    else:
        if template.stop_word is None:
            template.stop_word = tokenizer.eos_token
        stop_token_id = tokenizer.convert_tokens_to_ids(template.stop_word)

    history = []

    query = input('User：')
    # query = "请用简短的语言总结以下文本，并保留原文中出现的要点序号：1) 大功率快充电池散热方案已成熟，可有效解决散热问题\n大功率快充会带来发热量的大幅增加，高压电池包（Pack）的热管理至关重要。在电池包的安全设计上，可以通过应用隔热性能更高的隔热材料，例如陶瓷隔热垫、云母板，进行热扩散防护 ；在铜排金属零件表面粘贴绝缘材料（例如陶瓷复合带、云母纸）来防止高压打火，以此来提高电池包热扩散防护能力。目前业界已经有成熟的大功率快充电池热管理方案，可有效解决散热问题。以某车型的热管理为例，其水冷板设置在电池箱体下侧，可有效隔绝冷却液与模组，提高电池安全性。由于模组分布在两层，其水冷系统也分为上下两层，共 13 个冷却支路，每个冷却支路有两根水冷管并联，水冷管采用口琴管的方案，每根水冷管有 10 个并联通道。电池的液冷系统与整车的冷却系统是交互的，动力电池将热量传递给水冷板中的冷却液，冷却液再将热量通过热交换器传递给整车的冷却系统，最后将热量排放到空气中。考虑到快充效率和电池安全，在充电时，将电池包的温度控制在30℃左右，有效改善电池工作环境，提升充电安全性及寿命。"
    while True:
        query = query.strip()
        if query == "<|bye|>":
            history = []
            print("Start new session!")
        else:
            input_ids = build_prompt(tokenizer, template, query, copy.deepcopy(history), system=None).to(model.device)
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=stop_token_id
            )
            outputs = outputs.tolist()[0][len(input_ids[0]):]
            response = tokenizer.decode(outputs)
            response = response.strip().replace(template.stop_word, "").strip()
            # update history
            history.append({"role": 'user', 'message': query})
            history.append({"role": 'assistant', 'message': response})

            print("Assistant：{}".format(response))
        query = input('User：')
        # query = "你好"


if __name__ == '__main__':
    main()

