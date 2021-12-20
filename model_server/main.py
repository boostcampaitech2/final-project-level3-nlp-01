import uvicorn
from fastapi import FastAPI
from service import translate_zh_service, translate_en_service, translate_ko_service
from dto import ItemDto


app = FastAPI()


@app.post("/zh")
async def translate_zh(item: ItemDto):
    return translate_zh_service(item.text)


@app.post("/en")
async def translate_en(item: ItemDto):
    return translate_en_service(item.text)


@app.post("/ko")
async def translate_ko(item: ItemDto):
    return translate_ko_service(item.text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006, reload=False)
