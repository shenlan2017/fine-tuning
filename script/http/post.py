import json
import httpx
import argparse


def main():
    port_map = {"base": 2985, "sft": 2986}
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=["base", "sft"], required=True)
    args = parser.parse_args()

    url = f'http://127.0.0.1:{port_map[args.model_type]}/firefly'
    timeout = 60    # 超时设置

    # 生成超参数
    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0
    do_sample = True

    inputs = "请用简短的语言总结以下文本，并保留原文中出现的要点序号：1) 大功率快充电池散热方案已成熟，可有效解决散热问题\n大功率快充会带来发热量的大幅增加，高压电池包（Pack）的热管理至关重要。在电池包的安全设计上，可以通过应用隔热性能更高的隔热材料，例如陶瓷隔热垫、云母板，进行热扩散防护 ；在铜排金属零件表面粘贴绝缘材料（例如陶瓷复合带、云母纸）来防止高压打火，以此来提高电池包热扩散防护能力。目前业界已经有成熟的大功率快充电池热管理方案，可有效解决散热问题。以某车型的热管理为例，其水冷板设置在电池箱体下侧，可有效隔绝冷却液与模组，提高电池安全性。由于模组分布在两层，其水冷系统也分为上下两层，共 13 个冷却支路，每个冷却支路有两根水冷管并联，水冷管采用口琴管的方案，每根水冷管有 10 个并联通道。电池的液冷系统与整车的冷却系统是交互的，动力电池将热量传递给水冷板中的冷却液，冷却液再将热量通过热交换器传递给整车的冷却系统，最后将热量排放到空气中。考虑到快充效率和电池安全，在充电时，将电池包的温度控制在30℃左右，有效改善电池工作环境，提升充电安全性及寿命。"
  # 请求内容
    inputs = inputs.strip()

    params = {
        "inputs": inputs,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }

    timeout = httpx.Timeout(timeout)
    headers = {"Content-Type": "application/json", "Connection": "close"}
    session = httpx.Client(base_url="", headers=headers)
    response = session.request("POST", url, json=params, timeout=timeout)
    result = json.loads(response.text)['output']
    print(result)


if __name__ == '__main__':
    main()
