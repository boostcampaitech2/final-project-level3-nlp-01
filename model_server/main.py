import uvicorn
from fastapi import FastAPI

# 추후 모델 경량화가 진행되면 코드가 수정될 예정
class TranslateModel:
    def __init__(self, cfg):
        self.encoder_tokenizer = getattr(
            __import__("transformers"), cfg.encoder.tokenizer
        ).from_pretrained(
            enc_name
        )  # source lang
        self.decoder_tokenizer = getattr(
            __import__("transformers"), cfg.decoder.tokenizer
        ).from_pretrained(
            dec_name
        )  # target lang

        self.model = GrafomerModel(enc_name, dec_name, cfg)
        self.use_gpu = cfg.use_gpu
        if self.use_gpu:
            self.model.cuda()

    def predict(self, text):
        inputs = self.encoder_tokenizer(text, return_tensors="pt")
        if self.use_gpu:
            inputs = inputs.cuda()
        outputs = self.model.generate(
            inputs, max_length=40, num_beams=4, early_stopping=True
        )
        if self.use_gpu:
            outputs.cpu()
        outputs = self.decoder_tokenizer.decode(outputs[0])
        return outputs


model = TranslateModel(cfg)
app = FastAPI()


@app.post("/")
def translate(text: str):
    return model.predict(text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006, reload=True)
