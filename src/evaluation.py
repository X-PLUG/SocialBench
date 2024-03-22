import json
import os

from tqdm import tqdm

from src.baichuan_api import baichuan_turbo_call
from src.characterglm_api import characterglm_turbo_call
from src.dataset import RoleInteractDataset, format_predict, compute_score
from src.minimax_api import minimax_abab5_call, minimax_abab6_call
from src.openai_api import chatgpt_call, gpt4_call
from src.qwen_api import qwen_call, qwen_chat_call
from src.spark_api import spark_call
from src.utils import json_load, json_dump

CALL_FN = {
    "xingchen-plus": spark_call,
    "gpt-3.5": chatgpt_call,
    "gpt-4": gpt4_call,
    "baichuan-2-turbo": baichuan_turbo_call,
    # "baichuan-npc-turbo": baichuan_npc_call,
    "qwen-max": qwen_call,
    # "charglm-3": characterglm_call_v2,
    "glm-3-turbo": characterglm_turbo_call,
    "minimax-abab5.5s-chat": minimax_abab5_call,
    "minimax-abab6-chat": minimax_abab6_call,
    "qwen-72b-chat": qwen_chat_call
}


def compute_datalist_score(datalist):
    score = 0
    length = 0
    for data in datalist:
        if 'score' in data:
            length += 1
            score += data['score']
    acc = score / (length + 1e-12)
    return acc


def run(model: str, json_file: str, save_dir: str = '.'):
    assert model in CALL_FN
    os.makedirs(save_dir, exist_ok=True)
    dataset = RoleInteractDataset(json_file)
    save_name = os.path.split(json_file)[-1].replace(".json", "")
    datalist = []
    for data in tqdm(dataset):
        predict = CALL_FN[model](data['instruction'])
        label = json.loads(data['label'])
        meta = json.loads(data['meta'])
        print(label, "|", format_predict(predict), "|", predict)
        if predict is None:
            continue
        data['predict'] = predict
        data['score'] = compute_score(predict, label, meta)
        datalist.append(data)

    acc = compute_datalist_score(datalist)
    json_dump(datalist, os.path.join(save_dir, f'{save_name}_{model}_{round(acc, 5)}.json'))


def get_overall_results_memory(log_file: str):
    short_score = 0
    short_count = 0
    long_score = 0
    long_count = 0
    datalist = json_load(log_file)
    for data in datalist:
        if 'score' in data and len(data['predict']) > 0:
            if len(data['dialogue']) < 40:
                short_count += 1
                short_score += data['score']
            else:
                long_count += 1
                long_score += data['score']

    print(f"CM-Long |", round(long_score / long_count, 4))
    print(f"CM-Short |", round(short_score / short_count, 4))


def get_overall_results_emotion(log_file: str):
    situ_score = 0
    situ_count = 0
    emo_score = 0
    emo_count = 0
    datalist = json_load(log_file)
    for data in datalist:
        if 'predict' in data:
            answers = format_predict(data['predict'])
            if len(answers) > 0:
                score = compute_score(answers, data['label'])
                if data['meta']['category'] == "Individual-EP-SituationUnderstanding":
                    situ_count += 1
                    situ_score += score
                else:
                    emo_count += 1
                    emo_score += score

    print(f"EP-Emo. |", round(emo_score / emo_count, 4))
    print(f"EP-Situ. |", round(situ_score / situ_count, 4))


def get_overall_results_awareness(log_file: str):
    know_score = 0
    know_count = 0
    style_score = 0
    style_count = 0
    datalist = json_load(log_file)
    for data in datalist:
        if 'predict' in data:
            answers = format_predict(data['predict'])
            if len(answers) > 0:
                score = compute_score(answers, data['label'])
                if "RoleKnowledge" in data['meta']['category']:
                    know_count += 1
                    know_score += score
                else:
                    style_count += 1
                    style_score += score

    print(f"SA-Style |", round(style_score / style_count, 4))
    print(f"SA-Know. |", round(know_score / know_count, 4))


def get_overall_results_group(log_file: str):
    count_pos = 0
    count_neg = 0
    count_neu = 0
    score_pos = 0
    score_neg = 0
    score_neu = 0
    datalist = json_load(log_file)
    for data in datalist:
        if 'predict' in data:
            answers = format_predict(data['predict'])
            if len(answers) > 0:
                score = compute_score(answers, data['label'])
                if data['meta']['category'] == "Group-SAP-Positive":
                    count_pos += 1
                    score_pos += score
                elif data['meta']['category'] == "Group-SAP-Negative":
                    count_neg += 1
                    score_neg += score
                else:
                    count_neu += 1
                    score_neu += score

    print(f"Group-Pos |", round(score_pos / count_pos, 4), count_pos)
    print(f"Group-Neu |", round(score_neu / count_neu, 4), count_neu)
    print(f"Group-Neg |", round(score_neg / count_neg, 4), count_neg)
