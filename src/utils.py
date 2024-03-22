import json
import re
import csv
from typing import Callable


def format_name(name: str) -> str:
    return name.replace(" ", "_").replace(".txt", "").replace(".json", "")


def unformat_name(name: str) -> str:
    return name.replace("_", " ").replace(".txt", "").replace(".json", "")


def csv_load(f):
    with open(f, 'r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        data = list(reader)
    return data


def csv_dump(obj: list, f):
    with open(f, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(obj)


def json_dump(obj, f, indent=4, ensure_ascii=False):
    if str(f).endswith(".json"):
        with open(f, 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii))
    elif str(f).endswith(".jsonl"):
        with open(f, 'w', encoding='utf-8') as writer:
            assert type(obj) is list
            for data in obj:
                writer.write(json.dumps(data, ensure_ascii=ensure_ascii) + '\n')
    else:
        raise ValueError(f"Unexpected file type: {str(f)}")


def json_load(f):
    """Load a .json file into a dictionary."""
    if str(f).endswith(".json"):
        with open(f, 'r', encoding='utf-8') as reader:
            datalist = json.load(reader)
    elif str(f).endswith(".jsonl"):
        datalist = []
        with open(f, 'r', encoding='utf-8') as reader:
            for line in reader:
                datalist.append(json.loads(line))
    else:
        raise ValueError(f"Unexpected file type: {str(f)}")
    return datalist


def txt_load(f):
    s = ''
    with open(f, 'r', encoding='utf-8') as reader:
        for line in reader:
            s += line
    return s


def txt_dump(s: str, f):
    with open(f, 'w', encoding='utf-8') as writer:
        writer.write(s)


def format_data(data):
    s = '\n\n'
    for chat in data['conversations']:
        if 'User' in chat.keys():
            s += 'User: ' + chat['User'] + '\n'
        else:
            s += 'Assistant: ' + chat['Assistant'] + '\n'
    for c, a in data['choices'].items():
        s += f'{c}. ' + a + '\n'
    s += 'label: ' + data['label']
    return s


def get_accuracy(datalist):
    missing = 0
    hit = 0
    for data in datalist:
        if data['output'][0] in ['A', 'B', 'C', 'D']:
            if data['output'][0] == data['label']:
                hit += 1
        else:
            missing += 1
    acc = hit / (len(datalist) - missing)
    return acc, missing


def overlap(set1: set, set2: set):
    return len(set1 & set2) / len(set2)


def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(pattern.search(text))


def jaccard(_set1: set, _set2: set) -> float:
    return len(_set1 & _set2) / len(_set1 | _set2)


def deduplicate_texts(iterable: list, threshold: float = 0.8, key: Callable = None) -> list:
    results = []
    if key is None:
        def key(x):
            return x
    for i in range(len(iterable)):
        results.append(iterable[i])
        for j in range(i + 1, len(iterable)):
            sim = jaccard(set(key(iterable[i]).split(' ')), set(key(iterable[j]).split(' ')))
            if sim >= threshold:
                results.pop(-1)
                break

    return results
