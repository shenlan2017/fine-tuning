import argparse
from transformers import AutoTokenizer, AutoConfig, AddedToken
import torch
from loguru import logger
import copy
import os
import sys
sys.path.append("../../")
from component.utils import ModelUtils
from component.template import template_dict
import pathlib
from tqdm import tqdm
import json
import random


def get_inference_args():
    args = argparse.ArgumentParser("Inference Arguments")
    args.add_argument("--model_path", type=str, default="/mapping-data/qianli/firefly/output_summary_demo/qwen2-1.5b-sft-full")
    args.add_argument("--template_name", default="qwen")
    args.add_argument("--input_file", type=str, default="/mapping-data/qianli/firefly/data/summary/summary_test.jsonl")
    args.add_argument("--output_dir", type=str, default='/mapping-data/qianli/firefly/output_summary_demo/qwen2-1.5b-sft-full/full')
    args.add_argument("--tensor_dtype", type=str, default=None)
    args.add_argument("--use_cache", action="store_true")
    args.add_argument("--max_length", type=int, default=2048)
    args.add_argument("--do_sample", default=True)
    args.add_argument("--max_new_tokens", type=int, default=500)  # raw 0   bs_setting: 40
    args.add_argument("--top_p", type=float, default=0.9)  # raw 0.9  0.8  bs_setting: 0.9
    args.add_argument("--temperature", type=float, default=0.3)  # raw->0.3  # bs_setting: 0.9
    args.add_argument("--num_beams", type=int, default=1)  # bs_setting: 3
    args.add_argument("--length_penalty", type=float, default=None)  # bs_setting: 1
    args.add_argument("--repetition_penalty", type=float, default=1.0)  # bs_setting: 1.05
    args.add_argument("--min_length", type=int, default=None)
    args.add_argument("--no_repeat_ngram_size", type=int, default=None)


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
    # set params
    args = get_inference_args()
    template_name = args.template_name
    model_name_or_path = args.model_path

    adapter_name_or_path = None

    template = template_dict[template_name]
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    generation_kwargs = {
        "use_cache": args.use_cache,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "max_length": args.max_length,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "repetition_penalty": args.repetition_penalty,
        "min_length": args.min_length,
        "no_repeat_ngram_size": args.no_repeat_ngram_size
    }

    # 加载模型
    logger.info(f'Loading model from: {model_name_or_path}')
    logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = load_tokenizer(
        model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
    if template_name == 'chatglm2':
        stop_token_id = tokenizer.eos_token_id
    elif template_name == 'chatglm3':
        stop_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                         tokenizer.get_command("<|observation|>")]
    else:
        if template.stop_word is None:
            template.stop_word = tokenizer.eos_token
        stop_token_id = tokenizer.convert_tokens_to_ids(template.stop_word)

    prefix_list = ['给出下面这段文本的摘要，并保留原文中出现的要点序号：']

    # load inputs
    input_file = pathlib.Path(args.input_file).stem
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, input_file + "-res.jsonl")
    with open(args.input_file, "r", encoding="utf-8") as f, open(
            out_file, "w", encoding="utf-8"
    ) as fout:
        with torch.no_grad():

                for line in tqdm(f):
                    if args.input_file.endswith(".jsonl"):
                        line = json.loads(line)
                        tag = "src" if "src" in line else "query"
                        line = line[tag]
                    else:
                        line = str(line).strip()
                    prefix = random.choice(prefix_list)
                    line = f"{prefix}\n{line}"

                    input_ids = build_prompt(tokenizer, template, line,
                                             history=[], system=None).to(model.device)
                    if input_ids.shape[-1] > args.max_length:
                        input_ids = input_ids[..., :int(args.max_length // 1000 * 1000)]
                    output_ids = model.generate(
                        input_ids=input_ids,
                        eos_token_id=stop_token_id,
                        **generation_kwargs
                    )
                    output_ids = output_ids[0][len(input_ids[0]) :]
                    output_str = tokenizer.decode(output_ids)
                    output_str  = output_str.strip().replace(template.stop_word, "").strip()
                    fout.write(
                        json.dumps({"query": line, "reply": output_str}, ensure_ascii=False)
                        + "\n"
                    )
                    fout.flush()
                    print(f"output: {output_str}")


if __name__ == '__main__':
    main()