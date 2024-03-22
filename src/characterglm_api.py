import time

import zhipuai

from src.utils import unformat_name

api_key = "YOUR_API_KEY"
zhipuai.api_key = api_key

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


def create_meta(bot_name, bot_info) -> dict:
    bot_name = "Assistant" if bot_name is None else bot_name
    bot_info = "You are a helpful assistant." if bot_info is None else bot_info
    meta = dict(
        user_info="",
        bot_info=bot_info,
        bot_name=bot_name,
        user_name="User"
    )
    return meta


def create_prompt_for_open_domain(role_name: str, dialogue: list) -> list:
    prompt = []
    role_name = unformat_name(role_name)
    for utterance in dialogue:
        who = unformat_name(utterance['from'])
        if role_name != who:
            prompt.append(dict(role="user", content=f"{who}: {utterance['value']}"))
        else:
            prompt.append(dict(role="assistant", content=f"{utterance['value']}"))
    return prompt


def create_prompt_for_dialogue_emotion_detect(dialogue: list, choices: dict, lang="ZH") -> list:
    prompt = create_prompt_for_choices(role_name="", dialogue=dialogue, choices=choices, lang=lang)
    prompt.pop(-1)
    PROMPT = PROMPT_DIALOGUE_EMOTION_ZH if lang.lower() == "zh" else PROMPT_DIALOGUE_EMOTION_EN
    options = ""
    for choice, text in choices.items():
        options += f"{choice}. {text}\n"
    prompt.append(dict(role="user", content=PROMPT.format_map(
        {"options": options, "utterance": dialogue[-1]['value']}
    )))
    return prompt


def create_prompt_for_choices(role_name: str, dialogue: list, choices: dict, lang="ZH") -> list:
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


def characterglm_call_v2(meta: dict, prompt: str, model: str = "charglm-3"):
    messages = [{"role": "user", "content": prompt}]
    return characterglm_call(meta, messages, model)


def characterglm_call(meta: dict, prompt: list, model: str = "charglm-3"):
    response = zhipuai.model_api.invoke(
        model=model,
        meta=meta,
        prompt=prompt
    )
    try:
        return response['data']['choices'][0]['content']
    except KeyError:
        print(response)
        return ""


# "charglm-3"
def characterglm_turbo_call(prompt: str, model: str = "glm-3-turbo"):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=api_key)  # 填写您自己的APIKey
    response = None
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=model,  # 填写需要调用的模型名称
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except KeyError:
            print(response)
            time.sleep(5)
        except zhipuai.APITimeoutError:
            print(response)
            time.sleep(5)
        except zhipuai.APIRequestFailedError:
            print(response)
            time.sleep(5)
    return ""
