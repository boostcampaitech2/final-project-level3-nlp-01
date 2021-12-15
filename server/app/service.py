from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

translator = None


class TranslateModel:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def model_init():
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    global translator
    translator = TranslateModel(model, tokenizer)


def translate_service(language, text):
    return translator.translate(text)
