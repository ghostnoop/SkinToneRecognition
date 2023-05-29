import uvicorn
from fastapi import FastAPI

from models import Item
from services import predict_service
from skin_module.prediction_v2 import Prediction

app = FastAPI()
app.context = dict()


@app.on_event("startup")
async def startup_event():
    prediction = Prediction()
    app.context['prediction'] = prediction
    print("Startup event triggered")


@app.post("/predict")
async def predict_endpoint(item: Item):
    result = await predict_service(item, app.context['prediction'])
    print('-------')
    print(result)
    print(item.link)
    print('-------')
    return result


if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8585, workers=1)
