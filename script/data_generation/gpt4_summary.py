import pandas as pd
import time
from tqdm import tqdm
import requests
import argparse
import json
import os
from openai import OpenAI

SYS_INFO = {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        }


client = OpenAI(
    api_key="",  # 替换成真实DashScope的API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务endpoint
)


def get_json(file):
    if file.endswith("jsonl"):
        selected = []
        with open(file, 'r', encoding='utf-8') as f:  # os.path.join(base_dir, file)
            # lines=f.readlines()
            for line in tqdm(f):
                line = json.loads(line)  # .strip()
                selected.append(line)
    elif file.endswith("json"):
        with open(file, "r", encoding="utf-8") as f:
            selected = json.load(f)

    return selected


def get_reply_gpt_format_multi(query, history, add_sys=True):
    messages = []
    if add_sys:
        messages.append(SYS_INFO)
    if history is not None and len(history) > 0:
        for i, his in enumerate(history):
            if i % 2 == 0:
                messages.append({"role": "user", "content": his})
            else:
                messages.append({"role": "assistant", "content": his})
    messages.append({"role": "user", "content": query})
    completion = client.chat.completions.create(
        model="qwen-long",
        messages=messages,
        stream=True
    )

    res = []
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            res.append(chunk.choices[0].dict())

    return res


def filter_false(query, reply):
    query_use, query_false = [], []
    reply_use, reply_false = [], []
    for i in range(len(query)):
        if reply[i].get('status') == 200:
            query_use.append(query[i])
            reply_use.append(reply[i].get('response'))
        else:
            query_false.append(query[i])
            reply_false.append(reply[i].get('response'))

    return pd.DataFrame([query_use, reply_use]).transpose(), query_false, reply_false


def get_data(query, reply=None, fl=None):
    new_reply = []
    for i, q in tqdm(enumerate(query)):
        response = get_multi(q)
        if fl is not None:
            # fl.writelines("{}\t{}\t{}\n".format(q, str(response), str(response.get('reply'))))
            data = {"query": q, "reply": str(response.get('response')), "raw_resp": str(response)}
            fl.write(json.dumps(data, ensure_ascii=False) + '\n')
            fl.flush()
        new_reply.append(response)
        time.sleep(1)
    res, query_false, reply_false = filter_false(query, new_reply)

    return res, query_false, reply_false


def get_multi(query, history, success_code=200, w=2):
    # status = -1
    # while status != success_code:
    #     response = get_reply_gpt_format_multi(query, history)
    #     status = response.get('status')
    #     if status == success_code:
    #         return response.get('response')
    #
    #     print(f"Failed to get response, retrying in {w}s...")
    #     time.sleep(w)
    return get_reply_gpt_format_multi(query, history)

def main(input_file, output_file, data_column):
    if input_file.endswith("xlsx"):
        df = pd.read_excel(input_file)
        text_resources = df[data_column].tolist()
    elif input_file.endswith("csv"):
        df = pd.read_csv(input_file, sep=',', header=0, encoding='utf-8')
        text_resources = df[data_column].tolist()
    else:
        data = get_json(input_file)
        text_resources = []
        for item in data:
            text_resources.append(
                (item.get("src", item.get("query")), item.get("history"))
            )

    print(f'Total length: {len(text_resources)}')

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, (text, his) in tqdm(enumerate(text_resources), total=len(text_resources)):
            response = get_multi(text, his, w=5)  # get_summary  get_cot  get_translate
            f.write(
                json.dumps({"src": text, "tgt": response},
                           ensure_ascii=False) + "\n"
            )  # "total":s
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='input/raw_data.jsonl')
    parser.add_argument('--output_file', type=str, default='output/gen_inst.jsonl')
    parser.add_argument('--column_name', type=str, default='src')
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.column_name)
