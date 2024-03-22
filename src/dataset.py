import collections
import copy
import json
import re

from torch.utils.data import Dataset

from src.utils import json_load

PROMPT_EN = """
==Profile==
{role_profile}

==Conversations==
{conversations}

You are playing the role of {role_name}, you need to embody the knowledge and style of {role_name}.
Based on the provided role Profile and Conversations, please choose the best option (A, B, C, or D):
{options}

Your selection:
"""

PROMPT_ZH = """
==角色描述==
{role_profile}

==对话历史==
{conversations}

你要扮演{role_name}角色，你在聊天中要具备该角色对应的知识背景，语气风格等特征。
请根据所给的{role_name}角色描述和对话历史，从下面四个选项（A. B. C.和D.）中选择符合{role_name}的选项：
{options}

你的选择：
"""

PROMPT_DIALOGUE_EMOTION_EN = """
==Conversations==
{conversations}

Select the option () that best matches the mood in utterance "{utterance}". Single Choice
{options}

Your selection:
"""

PROMPT_DIALOGUE_EMOTION_ZH = """
==对话历史==
{conversations}

单选选择题，选择最符合"{utterance}"说话者当时心情的选项()
{options}

你的选择:
"""

PROMPT_OPEN_ZH = """
==角色描述==
{role_profile}

==对话历史==
{conversations}

你要扮演{role_name}角色，你在聊天中要具备该角色对应的知识背景，语气风格等特征。
请根据所给的{role_name}角色描述和对话历史，根据最后一个User的对话再补充一轮你作为Assistant的回复（一轮就好）：
Assistant: 
"""

PROMPT_OPEN_EN = """
==Profile==
{role_profile}

==Conversations==
{conversations}

You are playing the role of {role_name}, you need to embody the knowledge and style of {role_name}.
Based on the provided role Profile and Conversations, you must produce a reply as the Assistant to response to the latest User's message (one term is enough):
Assistant: 
"""


def format_question(dialogue, choices=None):
    conversations = ""
    for con in dialogue:
        role = con['from']
        text = con['value']
        conversations += f"{role}: {text}\n"

    options = ""
    if choices is not None:
        for choice, text in choices.items():
            options += f"{choice}. {text}\n"
    Output = collections.namedtuple('Output', ['dialogue', 'options'])
    return Output(dialogue=conversations, options=options)


def format_instruction(data):
    dialogue = data['dialogue']
    choices = data['choices'] if 'choices' in data else None
    category = data['meta']['category']
    lang = data['meta']['lang']
    outputs = format_question(dialogue, choices)
    if category == "Individual-MEM":
        PROMPT = PROMPT_OPEN_EN if lang.lower() == "en" else PROMPT_OPEN_ZH
        prompt = PROMPT.format_map({
            "role_profile": data['meta']['profile'][data['meta']['name']],
            "conversations": outputs.dialogue,
            "role_name": data['meta']['name'],
        })
    elif category == "Individual-EP-DialogueEmotionDetect":
        PROMPT = PROMPT_DIALOGUE_EMOTION_EN if lang.lower() == "en" else PROMPT_DIALOGUE_EMOTION_ZH
        prompt = PROMPT.format_map({
            "conversations": outputs.dialogue,
            "options": outputs.options,
            "utterance": dialogue[-1]["value"]
        })
    elif category in ["Individual-EP-HumorSarcasmDetect", "Individual-EP-SituationUnderstanding"]:
        prompt = f"{outputs.dialogue}\n{outputs.options}"
    else:
        assert category in [
            'Group-SAP-Positive',
            'Group-SAP-Negative',
            'Group-SAP-Neutral',
            'Individual-SA-RoleStyle',
            'Individual-SA-RoleKnowledge'
        ]
        PROMPT = PROMPT_EN if lang.lower() == "en" else PROMPT_ZH
        prompt = PROMPT.format_map({
            "role_profile": data['meta']['profile'][data['meta']['name']],
            "conversations": outputs.dialogue,
            "role_name": data['meta']['name'],
            "options": outputs.options
        })
    return prompt


class RoleInteractDataset(Dataset):
    def __init__(self, f: str, limit: int = None):
        self.datalist = json_load(f)
        if limit is not None:
            self.datalist = self.datalist[: limit]

    def __getitem__(self, i):
        data = copy.deepcopy(self.datalist[i])
        instruction = format_instruction(data)
        label = json.dumps(data['label'])  # type: str
        meta = json.dumps(data['meta'])  # type: str
        return dict(instruction=instruction, label=label, meta=meta)

    def __len__(self):
        return len(self.datalist)


def compute_score(predict: str, label: list, category: str = None):
    # labels = json.loads(label)  # type: list
    # category = json.loads(category)
    if category == "Individual-MEM":  # open-ended
        predict = predict.lower()
        if len(predict) == 0:
            return None
        score = 0
        for keyword in label:
            score += 1 if keyword.lower() in predict else 0
        return score / len(label)
    else:
        answers = format_predict(predict)
        if len(answers) == 0:
            return None
        if len(label) == 1:  # single-choice
            return 1 if answers[0] == label[0] else 0
        # multi-choices
        for answer in answers:
            if answer not in label:
                return 0
        return len(set(answers)) / len(set(label))


def format_predict(predict: str):
    if predict is None:
        return None
    answer = set()
    matches = re.findall(r'(\W+|^|[\u4e00-\u9fa5]+)([A-H])($|\W+|[\u4e00-\u9fa5]+)', predict)
    for match in matches:
        answer.add(match[1])
    return list(answer)
