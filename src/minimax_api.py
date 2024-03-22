import time

import requests

group_id = "YOUR_GROUP_ID"
api_key = "YOUR_API_KEY"

url = f"https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId={group_id}"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def minimax_abab5_call(s: str):
    return minimax_call(s, model="abab5.5s-chat")


def minimax_abab6_call(s: str):
    return minimax_call(s, model="abab6-chat")


def minimax_call(user_prompt: str, model: str = "abab5.5s-chat"):
    request_body = {
        # "model": "abab6-chat",
        "model": model,
        "tokens_to_generate": 1024,
        "reply_constraints": {"sender_type": "BOT", "sender_name": "MM智能助理"},
        "messages": [
            {'sender_type': 'USER', 'sender_name': 'user', 'text': user_prompt},
        ],
        "bot_setting": [
            {
                "bot_name": "MM智能助理",
                "content": "MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。",
            }
        ],
    }
    response = requests.post(url, headers=headers, json=request_body)
    status_code = response.status_code
    for i in range(5):
        try:
            if status_code == 200:
                reply = response.json()["reply"]
                if len(reply) == 0:
                    print("limit rate")
                    time.sleep(8)
                    continue
                return reply
            else:
                print(response._content)
                time.sleep(5)
        except KeyError:
            print(response)
            time.sleep(5)
            continue
    return ""


if __name__ == '__main__':
    print(minimax_call("1 + 122283 = ?"))
