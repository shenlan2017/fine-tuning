import argparse

import jsonlines
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

# def call_plugin(plugin_name: str, plugin_args: str) -> str:
#     if plugin_name == 'image_gen':
#         prompt = json5.loads(plugin_args)["prompt"]
#         prompt = urllib.parse.quote(prompt)
#         return json.dumps({'image_url': f'https://image.pollinations.ai/prompt/{prompt}'}, ensure_ascii=False)
#     elif plugin_name == '天气查询':
#         city = json5.loads(plugin_args)["city"]
#         city = urllib.parse.quote(city)
#         if not isinstance(city, str):
#             raise TypeError("City name must be a string")
#
#         api_key = "Sg3PPWFJS6prWTT7x"
#         url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={city}&language=zh-Hans&unit=c"
#         response = requests.get(url)
#         data = response.json()
#         if response.status_code == 200:
#             return json.dumps({'"temperature"': data["results"][0]["now"]["temperature"],
#                                "description": data["results"][0]["now"]["text"], }, ensure_ascii=False)
#         else:
#             raise Exception(f"Failed to retrieve weather: {response.status_code}")
#     elif plugin_name == 'get_lunar':
#         from datetime import date, datetime
#         from plunar_python import Lunar, Solar
#
#         solar_date = Solar.fromDate(datetime.now())
#         lunar_date = Lunar.fromDate(datetime.now())
#
#         festivals = ""
#         for festival in solar_date.getFestivals():
#             festivals += festival
#             festivals += "，"
#         for festival in solar_date.getOtherFestivals():
#             festivals += festival
#             festivals += "，"
#         festivals = festivals[:-1]
#
#         result = "solar date {}年{}月{}日星期{}，lunar date 农历{}年{}月{}, {}".format(solar_date.getYear(),
#                                                                                       solar_date.getMonth(),
#                                                                                       solar_date.getDay(),
#                                                                                       solar_date.getWeekInChinese(),
#                                                                                       lunar_date.getYearInGanZhi(),
#                                                                                       lunar_date.getMonthInChinese(),
#                                                                                       lunar_date.getDayInChinese(),
#                                                                                       festivals)
#
#         return result
#     elif plugin_name == 'car_controll':
#         response = "车辆控制已经{}".format(plugin_args)
#     else:
#         raise NotImplementedError
#
#
# tools = [
#     {
#         'name_for_human': '文生图',
#         'name_for_model': 'image_gen',
#         'description_for_model': '文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL',
#         'parameters': [
#             {
#                 'name': 'prompt',
#                 'description': '英文关键词，描述了希望图像具有什么内容',
#                 'required': True,
#                 'schema': {'type': 'string'},
#             }
#         ],
#     },
#
#     {
#         'name_for_human': '天气查询',
#         'name_for_model': 'get_current_weather',
#         'description_for_model': '天气查询工具通过调用天气API，获取给定城市的实时天气',
#         'parameters': [
#             {
#                 'name': 'city',
#                 'description': 'A city, in chinese',
#                 'required': True,
#                 'schema': {'type': 'string'},
#             }
#         ],
#     },
#     {
#         'name_for_human': '查询农历日期',
#         'name_for_model': 'get_lunar',
#         'description_for_model': '使用当前日期和时间，获取对应的公历和农历日期',
#         'parameters': [
#             {}
#         ],
#     },
#     {
#         'name_for_human': '车辆控制',
#         'name_for_model': 'car_control',
#         'description_for_model': '用于控制车辆组件',
#         'parameters': [
#             {
#                 'name': 'component',
#                 'description': 'the component need to be controlled',
#                 'required': True,
#                 'schema': {'type': 'string'},
#             },
#             {
#                 'name': 'command',
#                 'description': 'how to control the component',
#                 'required': True,
#                 'schema': {'type': 'string'},
#             }
#         ],
#     },
# ]


def get_inference_args():
    args = argparse.ArgumentParser("Inference Arguments")
    args.add_argument("--model_path", type=str, default="/mapping-data/qianli/firefly/output_tools_call_demo/qwen2-1.5b-sft-full-tools-1k-train-1800")
    args.add_argument("--template_name", default="qwen-tools")
    args.add_argument("--input_file", type=str, default="/mapping-data/qianli/firefly/data/tools/glaive_toolcall_zh_1k_test.jsonl")
    args.add_argument("--output_dir", type=str, default='/mapping-data/qianli/firefly/output_tools_call_demo/qwen2-1.5b-sft-full-tools-1k-train-1800/full')
    args.add_argument("--tensor_dtype", type=str, default=None)
    args.add_argument("--use_cache", action="store_true")
    args.add_argument("--max_length", type=int, default=1800)
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


def build_prompt(tokenizer, template, query, history, tools,system=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    function_format = template.function_format
    observation_format = template.observation_format
    system = system if system is not None else template.system
    system = system + str(tools)
    print(system)
    if template_name == 'chatglm2':
        prompt = tokenizer.build_prompt(query, history)
        input_ids = tokenizer.encode(prompt)
    elif template_name == 'chatglm3':
        input_ids = build_prompt_chatglm3(tokenizer, query, history, system)
    else:
        # history.append({"role": 'human', 'message': query})
        input_ids = []
        m = []
        # setting system information
        if system_format is not None:
            # system信息不为空
            if system is not None:
                system_text = system_format.format(content=system)
                input_ids = tokenizer.encode(system_text, add_special_tokens=False)
        # concat conversation
        for conv in history:
            # role, message = item['role'], item['message']
            message = conv['value']
            if conv['from'] == 'human':
                message = user_format.format(content=message, stop_token=tokenizer.eos_token)
                # print('human',message)
                m.append(message)
                tokens = tokenizer.encode(message, add_special_tokens=False)
                input_ids += tokens
            elif conv['from'] == 'gpt':
                message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
                # print('gpt',message)
                m.append(message)
                tokens = tokenizer.encode(message, add_special_tokens=False)
                input_ids += tokens
            else:

                tgt = conv['value']
                break



    input_ids = torch.tensor([input_ids], dtype=torch.long)
    print('m', m)
    return input_ids,tgt


def get_random_tools(tools_list, tool_to_exclude, num_samples=10):
    # 过滤掉与tool_to_exclude相同的工具
    filtered_tools = [tool for tool in tools_list if tool != tool_to_exclude]

    # 如果可用的工具不足9个，则返回所有剩余的工具
    if len(filtered_tools) < num_samples:
        return filtered_tools

    # 从过滤后的工具中随机抽取9个工具
    selected_tools = random.sample(filtered_tools, num_samples)

    return selected_tools
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
def extract_name_from_reply(reply):
    try:
        # 尝试将 reply 作为 JSON 解析
        reply_data = json.loads(reply)
        # 提取 name 字段
        return reply_data.get("name")
    except json.JSONDecodeError:
        # 如果 reply 不是 JSON 格式，返回 None 或者其他默认值
        return None

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


    # load inputs
    input_file = pathlib.Path(args.input_file).stem

    unique_tools = set()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading data"):
            obj = json.loads(line.strip())

            unique_tools.add(obj['tools'])

    # 将集合中的字符串解析为对象并存入列表
    unique_tools_list = [json.loads(tool) for tool in unique_tools]
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, input_file + "-res.jsonl")
    acc = 0
    lens = 0
    with open(args.input_file, "r", encoding="utf-8") as f, open(
            out_file, "w", encoding="utf-8"
    ) as fout:
        with torch.no_grad():

                for line in tqdm(f):
                    if args.input_file.endswith(".jsonl"):
                        lines = json.loads(line)
                        line = lines['conversations']
                        tools = lines['tools']
                    else:
                        line = str(line).strip()
                    #测试集如果没有涉及工具调用，则不进行推理
                    if tools !='[]':
                        # prefix = random.choice(prefix_list)
                        # line = f"{prefix}\n{line}"
                        new_tools_list = get_random_tools(unique_tools_list, tools)
                        if tools not in new_tools_list:
                            new_tools_list.append(tools)
                        random.shuffle(new_tools_list)

                        input_ids,tgt = build_prompt(tokenizer, template, line,
                                                 history=line, tools=new_tools_list,system=None)
                        input_ids = input_ids.to(model.device)
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
                        lens+=1
                        extracted_name = extract_name_from_reply(output_str)
                        tgt = tgt.replace("'", '"')
                        tgt = json.loads(tgt)
                        if extracted_name == tgt['name']:
                            acc+=1
                        else:
                            print("error",extracted_name)
                        fout.write(
                            json.dumps({"query": line, "tools":new_tools_list, "reply": output_str}, ensure_ascii=False)
                            + "\n"
                        )
                        fout.flush()
                        print(f"output: {output_str}")
                    else:
                        continue
        print('acc:',acc/lens)
        fout.write(
            json.dumps({"ACC:":acc/lens}, ensure_ascii=False)
            + "\n"
        )

if __name__ == '__main__':
    main()