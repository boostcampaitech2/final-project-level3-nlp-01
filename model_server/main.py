import uvicorn
from fastapi import FastAPI
from service import translate_service
from dto import ItemDto


app = FastAPI()


@app.post("/")
async def translate(item: ItemDto):
    return translate_service(item.text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006, reload=False)
