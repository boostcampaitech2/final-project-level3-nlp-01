from util import model_init

predict = model_init()


def translate_service(text):
    return predict(text)
