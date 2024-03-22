# -*- coding: utf-8 -*-
import json
import time

import requests

X_AK = "YOUR_API_KEY"


# gpt-3.5-turbo | gpt-4-turbo-128k | gpt-4 8K
def invoke_idealab(inputs: dict):
    base_url = "https://idealab.alibaba-inc.com/aigc/v1/askTextToTextMsg"

    res = None
    while True:
        try:
            res = requests.post(
                base_url,
                headers={
                    'X-AK': X_AK,
                    'Content-Type': 'application/json',
                    'Accept-Encoding': 'utf-8'
                },
                data=json.dumps(inputs)
            )
            break
        except requests.exceptions.ConnectionError:
            time.sleep(5)
            print('Connection error, retrying ......')
            continue

    status_code = res.status_code
    result = None
    if status_code == 200:
        try:
            result = json.loads(res.text)
        except json.decoder.JSONDecodeError:
            print(res)
            exit(-1)
    else:
        raise RuntimeError(res.text)
    return result


def invoke_openai(inputs: str, model: int = 0, maxTokens: int = None) -> str:
    """ model: 0: gpt-3.5-turbo | 1: gpt-4-turbo-128k | 2: gpt-4 8K """
    model = ['gpt-3.5-turbo', 'gpt-4-turbo-128k', 'gpt-4 8K'][model]
    if maxTokens is None:
        output = invoke_idealab({
            "model": model,
            "prompt": inputs
        })
    else:
        output = invoke_idealab({
            "model": model,
            "prompt": inputs,
            "maxTokens": maxTokens
        })
    if output['data'] is None:
        raise RuntimeError(output)
    for _ in range(3):
        try:
            return output['data']['content']
        except KeyError:
            if 'message' in output and "context_length_exceeded" in output['message']:
                return ""
            time.sleep(1)
            print("Retrying ...")
            pass
    print(output)
    raise RuntimeError


def chatgpt_call(inputs: str):
    return invoke_openai(inputs, model=0)


def gpt4_call(inputs: str):
    return invoke_openai(inputs, model=1)


def query_balance(businessCode='tongyixingchen1213'):
    base_url = "https://idealab.alibaba-inc.com/aigc/v1/queryBalance"

    inputs = {
        "businessCode": businessCode,
        "sceneCode": 'default'
    }

    res = requests.post(
        base_url,
        headers={
            'X-AK': X_AK,
            'Content-Type': 'application/json', 'Accept-Encoding': 'utf-8'
        },
        data=json.dumps(inputs)
    )

    status_code = res.status_code
    if status_code == 200:
        result = json.loads(res.text)
    else:
        result = None
    return result


def query_bills(date_start='2023-01-01', date_end=None):
    base_url = "https://idealab.alibaba-inc.com/aigc/v1/queryBills"

    if date_end is None:
        from datetime import datetime
        # 获取当前日期和时间
        now = datetime.now()
        # 格式化输出日期为"YYYY-MM-DD"格式
        date_end = now.strftime("%Y-%m-%d")

    inputs = {
        "dateStart": date_start,
        "dateEnd": date_end
    }

    res = requests.post(
        base_url,
        headers={
            'X-AK': X_AK,
            'Content-Type': 'application/json', 'Accept-Encoding': 'utf-8'
        },
        data=json.dumps(inputs)
    )

    status_code = res.status_code
    if status_code == 200:
        result = json.loads(res.text)
    else:
        result = None
    return result


if __name__ == '__main__':
    # print(invoke_openai("hello", model=0))
    print(query_bills('2024-03-17'))
