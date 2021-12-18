import requests
import json
import yaml

lang = {"한국어": "url", "english": "url", "中文": "url"}


def load_yaml():
    global lang
    with open("config.yaml", "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    lang["한국어"] = conf["kr_url"]
    lang["english"] = conf["en_url"]
    lang["中文"] = conf["zh_url"]


def translate_service(language, text):
    data = {"text": text}
    res = requests.post(lang[language], data=json.dumps(data))
    return res.text
