import json
import argparse


def extract_name_from_reply(reply_str):
    """
    从reply字段中提取name字段的值
    """
    try:
        reply_json = json.loads(reply_str)
        return reply_json.get('name', None)
    except json.JSONDecodeError:
        return None


def extract_tools_names(tools_list):
    """
    从tools列表中提取每个工具的name字段
    """
    extracted_tools = []
    for tool in tools_list:
        try:
            # 处理每个工具条目的JSON解码
            if isinstance(tool, str):
                tool_dict = json.loads(tool.replace("'", '"'))  # 确保JSON字符串中使用双引号
            else:
                tool_dict = tool
            extracted_tools.append(tool_dict)
        except (json.JSONDecodeError, TypeError):
            continue
    return extracted_tools


def calculate_accuracy(input_file):
    acc = 0
    total = 0
    

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get('reply'):
                reply_name = extract_name_from_reply(obj.get('reply', ''))
                tools_names = extract_tools_names(obj.get('tools', []))


                for tool in tools_names:
                    if not tool: # 有空集出现的情况
                        continue
                    if reply_name and tool[0].get('name'):
                        if reply_name == tool[0]['name']:
                            acc += 1
                            break
                total += 1
    return acc, total


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy of tool calls in JSONL file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    args = parser.parse_args()  
    acc, total = calculate_accuracy(args.input_file)

    accuracy = acc / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({acc}/{total})")


if __name__ == "__main__":
    main()
