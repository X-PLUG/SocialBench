import time
from http import HTTPStatus

from dashscope import Generation
from urllib3.exceptions import MaxRetryError, NewConnectionError

PRODUCT_API_KEY = "YOUR_API_KEY"

PROMPT = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""


def spark_call(user_prompt, system_prompt=""):
    prompt = PROMPT.format_map({"user_prompt": user_prompt, "system_prompt": system_prompt})
    for _ in range(5):
        try:
            responses = Generation.call(
                model='qwen-spark-plus',
                prompt=prompt,
                api_key=PRODUCT_API_KEY,
                use_raw_prompt=True,
                seed=1683806810
            )

            if responses.status_code == HTTPStatus.OK:
                return responses.output['text']
            else:
                print('Failed request_id: %s, status_code: %s, code: %s, message:%s' %
                      (responses.request_id, responses.status_code, responses.code,
                       responses.message))
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


if __name__ == '__main__':
    r = spark_call("你好")
    print(r)
