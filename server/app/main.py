import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/{language}")
async def translate(language: str, text: str):
    return {"language": language}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
