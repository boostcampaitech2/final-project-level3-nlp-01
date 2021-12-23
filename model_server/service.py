from util import model_init

predict_zh = model_init("zh")
predict_en = model_init("en")
predict_ko = model_init("ko")


def translate_zh_service(text):
    generate_text = predict_zh(text)
    return generate_text


def translate_en_service(text):
    generate_text = predict_en(text)
    return generate_text


def translate_ko_service(text):
    generate_text = predict_ko(text)
    return generate_text
