import random
import time
from http import HTTPStatus

import dashscope
from urllib3.exceptions import MaxRetryError, NewConnectionError

dashscope.api_key = "YOUR_API_KEY"


def call_with_messages():
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '如何做西红柿炒鸡蛋？'}]
    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_max,
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        seed=random.randint(1, 10000),
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


# 'qwen-72b-chat' 'qwen-max-longcontext' 'qwen-max'
def qwen_call(user_prompt, system_prompt='You are a helpful assistant.', model='qwen-max'):
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}]
    for _ in range(5):
        try:
            response = dashscope.Generation.call(
                model,
                messages=messages,
                # set the random seed, optional, default to 1234 if not set
                seed=random.randint(1, 10000),
                result_format='message',  # set the result to be "message" format.
            )
            if response.status_code == HTTPStatus.OK:
                return response.output["choices"][0]["message"]["content"]
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
                time.sleep(5)
        except ConnectionError:
            time.sleep(5)
            continue
        except MaxRetryError:
            time.sleep(5)
            continue
        except NewConnectionError:
            time.sleep(5)
            continue
        except TimeoutError:
            time.sleep(5)
            continue
    return ""


def qwen_chat_call(user_prompt, system_prompt='You are a helpful assistant.'):
    return qwen_call(user_prompt, system_prompt, model='qwen-72b-chat')


if __name__ == '__main__':
    print(qwen_chat_call("你好"))
