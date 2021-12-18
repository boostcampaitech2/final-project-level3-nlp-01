import requests
import json

lang = {"한국어": "url", "english": "url", "中文": "url"}


def translate_service(language, text):
    data = {"text": text}
    res = requests.post(lang[language], data=json.dumps(data))
    return res.text
