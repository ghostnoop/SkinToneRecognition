import aiohttp

from models import Item
from skin_module.prediction import Prediction


async def predict_service(item: Item, prediction: Prediction):
    content = None
    async with aiohttp.ClientSession() as session:
        async with session.get(item.link) as response:
            content = await response.read()
    if content:
        return prediction.predict(content, item.coordinates)
