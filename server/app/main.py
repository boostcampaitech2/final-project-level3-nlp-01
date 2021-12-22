import uvicorn
from fastapi import FastAPI

from core.dto import ItemDto
from service import translate_service, load_yaml

load_yaml()
app = FastAPI()


@app.post("/translate")
async def translate(item: ItemDto):
    trans_text = translate_service(item.target_lang, item.text)
    return trans_text


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
