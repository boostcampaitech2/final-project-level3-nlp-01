from util import model_init

predict = model_init("zh")


def translate_service(text):
    return predict(text)
