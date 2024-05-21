import json
import time

import requests
from urllib3.exceptions import MaxRetryError, NewConnectionError

from src.utils import unformat_name

url = "https://api.baichuan-ai.com/v1/chat/completions"
api_key = "YOUR_API_KEY"


PROMPT_ZH = """
从下面四个选项（A. B. C.和D.）中选择符合{role_name}的最佳选项：
{options}
请用字母(A. B. C. 或D.)开头写出你的答案：
"""

PROMPT_EN = """
Please select the most appropriate option for {role_name} from the following four choices (A. B. C. and D.):
{options}
Please begin your answer with the letter (A. B. C. or D.):
"""


PROMPT_DIALOGUE_EMOTION_EN = """
Select the option () that best matches the mood in utterance "{utterance}". Single Choice
{options}

Your selection, begin with the letter:
"""

PROMPT_DIALOGUE_EMOTION_ZH = """
单选选择题，选择最符合"{utterance}"说话者当时心情的选项()
{options}

你的选择，请以选项字母开头:
"""


def create_character_profile(character_name: str = None, character_info: str = None) -> dict:
    character_name = "Assistant" if character_name is None else character_name
    character_info = "You are a helpful assistant." if character_info is None else character_info
    character_profile = dict(
        character_name=character_name,
        character_info=character_info,
        user_name="User",
        user_info="User"
    )
    return character_profile


def create_messages_for_dialogue_emotion_detect(dialogue: list, choices: dict, lang="ZH") -> list:
    prompt = create_messages_for_choices(role_name="", dialogue=dialogue, choices=choices, lang=lang)
    prompt.pop(-1)
    PROMPT = PROMPT_DIALOGUE_EMOTION_ZH if lang.lower() == "zh" else PROMPT_DIALOGUE_EMOTION_EN
    options = ""
    for choice, text in choices.items():
        options += f"{choice}. {text}\n"
    prompt.append(dict(role="user", content=PROMPT.format_map(
        {"options": options, "utterance": dialogue[-1]['value']}
    )))
    return prompt


def create_messages_for_open_domain(role_name: str, dialogue: list) -> list:
    prompt = []
    role_name = unformat_name(role_name)
    for utterance in dialogue:
        who = unformat_name(utterance['from'])
        if role_name != who:
            prompt.append(dict(role="user", content=f"{who}: {utterance['value']}"))
        else:
            prompt.append(dict(role="assistant", content=f"{utterance['value']}"))
    return prompt


def create_messages_for_choices(role_name: str, dialogue: list, choices: dict, lang="ZH") -> list:
    prompt = []
    role_name = unformat_name(role_name)
    for utterance in dialogue:
        who = unformat_name(utterance['from'])
        if role_name != who:
            prompt.append(dict(role="user", content=f"{who}: {utterance['value']}"))
        else:
            prompt.append(dict(role="assistant", content=f"{utterance['value']}"))
    options = ""
    for choice, text in choices.items():
        options += f"{choice}. {text}\n"
    PROMPT = PROMPT_ZH if lang.lower() == "zh" else PROMPT_EN
    prompt.append(dict(role="user", content=PROMPT.format_map(
        {"options": options, "role_name": role_name}
    )))
    return prompt


def do_request():
    data = {
        "model": "Baichuan-NPC-Turbo",
        "character_profile": {
            "character_name": "大罗",
            "character_info": "角色基本信息：大罗被广泛认为是有史以来最伟大的足球运动员之一。因为其强悍恐怖的攻击力被冠以“外星人”称号。大罗曾三度当选世界足球先生、两度获得金球奖，为巴西夺得两次世界杯冠军及一次亚军。效力过皇家马德里，巴塞罗那，AC米兰，国际米兰等豪门俱乐部，进球无数。",
            "user_name": "小乐",
            "user_info": "某体育频道解说员，在中国举办的大罗球迷见面会上做为主持人"
        },
        "messages": [
            {
                "role": "user",
                "content": "你喜欢那个球星吗"
            },
            {
                "role": "assistant",
                "content": "梅西"
            },
            {
                "role": "user",
                "content": "他是哪里人"
            },
        ],
        "temperature": 0.8,
        "top_k": 10,
        "max_tokens": 3600,
        "stream": False
    }

    json_data = json.dumps(data)

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key
    }

    response = requests.post(url, data=json_data, headers=headers, timeout=60)

    a = response._content
    print(type(a))
    print(a.decode('utf-8'))
    res = json.loads(a.decode('utf-8'))
    return res["choices"][0]["message"]["content"]


def baichuan_npc_call(character_profile: dict, prompt: str, model: str = "Baichuan-NPC-Turbo"):
    messages = [{"role": "user", "content": prompt}]
    return baichuan_call(character_profile, messages, model)


def baichuan_call(character_profile: dict, messages: list, model: str = "Baichuan-NPC-Turbo"):
    data = {
        "model": model,
        "character_profile": character_profile,
        "messages": messages,
        "temperature": 0.8,
        "top_k": 10,
        "max_tokens": 3072,
        "stream": False
    }

    json_data = json.dumps(data)

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key
    }

    for i in range(5):
        res = None
        try:
            response = requests.post(url, data=json_data, headers=headers, timeout=60)
            res = response._content.decode('utf-8')
            res = json.loads(res)
            return res["choices"][0]["message"]["content"]
        except KeyError:
            if res is not None and 'error' in res:
                if "Internal Server Error" in res['error']['message']:
                    return ""
            print(res)
            time.sleep(1)
    return ""


def baichuan_turbo_call(prompt: str):
    data = {
        "model": "Baichuan2-Turbo",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    json_data = json.dumps(data)

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key
    }

    for i in range(5):
        res = None
        try:
            res = requests.post(url, data=json_data, headers=headers, timeout=60)
            res = res._content.decode('utf-8')
            res = json.loads(res)
            return res["choices"][0]["message"]["content"]
        except KeyError:
            print(res)
            time.sleep(1)
            continue
        except ConnectionError:
            time.sleep(5)
            continue
        except MaxRetryError:
            time.sleep(5)
            continue
        except NewConnectionError:
            time.sleep(5)
            continue
    return ""


if __name__ == '__main__':
    do_request()
