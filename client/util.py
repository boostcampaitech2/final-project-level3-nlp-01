import requests
import json


def translate(text, target):
    # requests.post 와 target을 이용하여 모델에게 request를 보내는 코드 작성
    data = {"text": text, "target_lang": target}
    res = requests.post("http://0.0.0.0:8000/translate", data=json.dumps(data))
    return res.text
