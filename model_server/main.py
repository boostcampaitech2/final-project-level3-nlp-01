import uvicorn
from fastapi import FastAPI
from service import predict_func


model_predict = predict_func()
app = FastAPI()


@app.post("/")
def translate(text: str):
    return model_predict(text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006, reload=True)
