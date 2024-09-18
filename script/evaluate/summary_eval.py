import pandas as pd
import json
import os
import numpy as np
import math
import re
import jsonlines
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu


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


def write_jsonl(fn, res):
    with jsonlines.open(fn, "w") as wfd:
        for item in res:
            try:
                wfd.write(item)
            except Exception as e:
                print(e)
                print(item)


def output_excel(df, out_fn):
    df.to_excel(out_fn, encoding='GB18030', index=True)


def calc_rougel():
    tgt_r = [' '.join(list(sen)) for sen in tgt]  # RAG:tgt_res:tgt_res  xiaoai:tgt
    print(tgt_r[0])
    model_reply = "pytorch_model.bin"  # reply
    model_query = "input"  # query

    for fn in fn_lst:
        res = get_json(f"{base_dir}/{fn}/xiaoai_summary-res.jsonl")
        # res = get_json(f"{base_dir}/{fn}/RAG_test-res.jsonl")  # xiaoai_summary-res  RAG_test-res
        # xiaoai_summary-res  RAG_test-res
        res_df = pd.DataFrame(res)
        res_lst = res_df[model_reply].values.tolist()[100:]
        res_lst_r = [' '.join(list(sen)) for sen in res_lst]

        res_qlst = res_df[model_query].values.tolist()[100:]
        # print(res_lst[:100])
        # break
        rouger = Rouge()

        cnt = 0
        scores = []
        for p, t in tqdm(zip(res_lst_r, tgt_r), total=len(tgt_r)):
            try:
                score = rouger.get_scores(p, t, avg=False)[0]
                # tmp_f = scores.get('rouge-l').get('f')
                scores.append(score)
            except:
                print('error!')
                # print(f'p: {p}')
                # print(f't: {t}')
                print('continue...')
        # break
        print("rouge calc done! postprocessing...")
        scores_mean = {}
        for k, _ in scores[0]['rouge-l'].items():
            scores_mean[k] = np.mean([item['rouge-l'][k] for item in scores])

        #     scores = rouger.get_scores(res_lst_r, tgt_r, avg=False)
        rougel_lst = [item.get("rouge-l").get("f") for item in scores]

        #     scores_mean = rouger.get_scores(res_lst_r, tgt_r, avg=True)
        print(f"fn: {fn}\nvalid samples: {len(scores)}")
        print(f"fn: {fn}\navg rouge_score: {scores_mean}")

        # out_df = pd.DataFrame({"query":src, "input":res_qlst, "output": res_lst, "golden":tgt, "rouge-l":rougel_lst})
        # break
        # output_excel(out_df, f'0409summary/res/rouge-l_res/{fn}.xls


def calc_bleu():
    tgt_r = [list(sen) for sen in tgt_res]  # RAG:tgt_res:tgt_res  xiaoai:tgt
    print(tgt_r[0])
    # model_reply = "pytorch_model.bin"
    # model_query = "input"
    model_reply = "reply"
    model_query = "query"
    # model_reply = "src"
    # model_query = "tgt"

    for fn in fn_lst:
        # res = get_json(f"{base_dir}/{fn}/xiaoai_summary-res.jsonl")
        # res = get_json(f"{base_dir}/{fn}/RAG_test-res.jsonl")
        res = get_json(f"{base_dir}/{fn}/RAG_human-res.jsonl")
        # xiaoai_summary-res  RAG_test-res
        res_df = pd.DataFrame(res)
        # res_lst = res_df[model_reply].values.tolist()[:100]
        res_lst = res_df[model_reply].values.tolist()[:250]

        res_lst_r = [list(sen) for sen in res_lst]

        # res_qlst = res_df[model_query].values.tolist()[:100]
        res_qlst = res_df[model_query].values.tolist()[:250]

        # print(res_lst[:100])
        # break
        weights = [(1. / 2., 1. / 2.), (1. / 3., 1. / 3., 1. / 3.),
                   (0.25, 0.25, 1. / 4., 1. / 4.)]

        scores = []
        for p, t in tqdm(zip(res_lst_r, tgt_r), total=len(tgt_r)):
            try:
                # print(p)
                # print(t)
                scores.append(sentence_bleu([t], p, weights=weights))
                # print(sentence_bleu(t, p, weights=weights))
                # break
            except:
                print('error!')
                print(f'p: {p}')
                print(f't: {t}')
                print('continue...')
                # break
        #     print("bleu calc done! postprocessing...")
        #     scores_mean = {}
        #     for k,_ in scores[0]['rouge-l'].items():
        #         scores_mean[k] = np.mean([item['rouge-l'][k] for item in scores])

        scores = np.array(scores)
        scores_mean = np.mean(scores, axis=0)

        # #     scores_mean = rouger.get_scores(res_lst_r, tgt_r, avg=True)
        print(f"fn: {fn}\nvalid samples: {len(scores)}")
        print(f"fn: {fn}\nmacro blue_score2/3/4: {scores_mean}")

        # out_df = pd.DataFrame({"query":src, "input":res_qlst, "output": res_lst, "golden":tgt, "rouge-l":rougel_lst})
        # break
        # output_excel(out_df, f'0409summary/res/rouge-l_res/{fn}.xlsx')
        # output_excel(out_df, f'0409summary/res/xiaoai/{fn}.xlsx')


def main():
    pass


if __name__ == "__main__":
    main()