import json
import jsonlines
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse

def load_jsonl(file_path):
    """Load a JSONL file and return a list of records."""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def calculate_bleu(reference, hypothesis):
    """Calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores between reference and hypothesis."""
    chencherry = SmoothingFunction()
    reference = [list(reference)]  # BLEU expects a list of reference sequences
    hypothesis = list(hypothesis)

    bleu_1 = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
    bleu_2 = sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
    bleu_3 = sentence_bleu(reference, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
    bleu_4 = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)

    return bleu_1, bleu_2, bleu_3, bleu_4

def calculate_rouge(reference, hypothesis):
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores between reference and hypothesis."""
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    rouge_1_f1 = scores['rouge-1']['f']
    rouge_2_f1 = scores['rouge-2']['f']
    rouge_l_f1 = scores['rouge-l']['f']
    return rouge_1_f1, rouge_2_f1, rouge_l_f1

def main(ground_truth_file, generated_file):
    # Load the data
    ground_truth_data = load_jsonl(ground_truth_file)
    generated_data = load_jsonl(generated_file)

    # Ensure both files have the same length
    assert len(ground_truth_data) == len(generated_data), "The number of records in the files do not match."

    # Initialize lists to hold scores
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    # Calculate BLEU and ROUGE scores for each pair of reference and hypothesis
    for ground, gen in tqdm(zip(ground_truth_data, generated_data), total=len(ground_truth_data)):
        reference = ground['reply']
        hypothesis = gen['reply']

        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(reference, hypothesis)
        rouge_1, rouge_2, rouge_l = calculate_rouge(reference, hypothesis)

        bleu_1_scores.append(bleu_1)
        bleu_2_scores.append(bleu_2)
        bleu_3_scores.append(bleu_3)
        bleu_4_scores.append(bleu_4)
        rouge_1_scores.append(rouge_1)
        rouge_2_scores.append(rouge_2)
        rouge_l_scores.append(rouge_l)

    # Calculate average scores
    avg_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores)
    avg_bleu_2 = sum(bleu_2_scores) / len(bleu_2_scores)
    avg_bleu_3 = sum(bleu_3_scores) / len(bleu_3_scores)
    avg_bleu_4 = sum(bleu_4_scores) / len(bleu_4_scores)
    avg_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores)
    avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    # Print the results
    print(f"Average BLEU-1 score: {avg_bleu_1}")
    print(f"Average BLEU-2 score: {avg_bleu_2}")
    print(f"Average BLEU-3 score: {avg_bleu_3}")
    print(f"Average BLEU-4 score: {avg_bleu_4}")
    print(f"Average ROUGE-1 score: {avg_rouge_1}")
    print(f"Average ROUGE-2 score: {avg_rouge_2}")
    print(f"Average ROUGE-L score: {avg_rouge_l}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BLEU and ROUGE scores for generated text.")
    parser.add_argument('--ground_truth_file', type=str, required=True, help="Path to the ground truth JSONL file")
    parser.add_argument('--generated_file', type=str, required=True, help="Path to the generated JSONL file")
    args = parser.parse_args()

    main(args.ground_truth_file, args.generated_file)
