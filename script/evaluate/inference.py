import argparse
import json
from loguru import logger
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import jsonlines
from tqdm import tqdm

from component.collator import SFTDataCollator
from component.dataset import ChatGLM2SFTDataset, ChatGLM3SFTDataset, UnifiedSFTDataset
from component.template import template_dict

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_args_file", type=str, default='/mapping-data/qianli/firefly/data/sum_train_2048.jsonl',
                        help="Path to evaluation arguments file")
    parser.add_argument("--local_rank", type=int, help="")
    parser.add_argument("--model_name_or_path", type=str,
                        default='/mapping-data/qianli/firefly/output_summary_demo/qwen2-1.5b-sft-full',
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str,
                        default='/mapping-data/qianli/firefly/output_summary_demo/qwen2-1.5b-sft-full/outputs',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--template_name", type=str, default='qwen', help="The name of the template to use.")

    training_args = parser.parse_args()

    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'eval.log'))
    logger.info("Eval arguments:{}".format(training_args))

    # 设置随机种子
    set_seed(0)

    return training_args


def load_model_and_tokenizer(model_name_or_path, device):
    """
    Load the pre-trained model and tokenizer.
    """
    logger.info(f'Loading model and tokenizer from {model_name_or_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.to(device)
    return model, tokenizer


def evaluate_model(model, tokenizer, dataloader, output_file, device='cpu'):
    """
    Evaluate the model by generating text and saving the results.
    """
    model.eval()

    with jsonlines.open(output_file, 'w') as writer:
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            output_ids = model.generate(input_ids=input_ids)

            generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for i, gen_text in enumerate(generated_text):
                result = {
                    'src': tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
                    'tgt': gen_text,
                }
                writer.write(result)
                print(result)

        logger.info(f'Evaluation completed. Results saved to {output_file}')


def load_sft_dataset(args, tokenizer):
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]
    if 'chatglm2' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM2SFTDataset')
        train_dataset = ChatGLM2SFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    elif 'chatglm3' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM3SFTDataset')
        train_dataset = ChatGLM3SFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    else:
        logger.info('Loading data with UnifiedSFTDataset')
        train_dataset = UnifiedSFTDataset(args.eval_args_file, tokenizer, args.max_seq_length, template)
    return train_dataset


def main():
    # Setup configurations and seed
    training_args = setup_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(training_args.model_name_or_path, device)
    eval_dataset = load_sft_dataset(training_args, tokenizer)

    # Create DataLoader for evaluation
    data_collator = SFTDataCollator(tokenizer, training_args.max_seq_length)
    dataloader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size,
                            collate_fn=data_collator,shuffle=False)

    # Evaluate the model and save generated text
    output_file = join(training_args.output_dir, 'eval_results.jsonl')
    evaluate_model(model, tokenizer, dataloader, output_file, device=device)


if __name__ == "__main__":
    main()
