import uvicorn
from fastapi import FastAPI

from models import Item
from services import predict_service
from skin_module.prediction import Prediction

app = FastAPI()
app.context = dict()


@app.on_event("startup")
async def startup_event():
    prediction = Prediction()
    app.context['prediction'] = prediction
    print("Startup event triggered")


@app.post("/predict")
async def predict_endpoint(item: Item):
    return await predict_service(item, app.context['prediction'])


if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8585, workers=10)
