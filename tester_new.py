import time

import requests

from skin_module.prediction_v2 import Prediction


def worker():
    prediction = Prediction()
    st = time.monotonic()

    content = requests.get('https://www.tprteaching.com/wp-content/uploads/2022/12/peoples-or-peoples.jpg').content

    result = prediction._predict(content)
    print(result)

if __name__ == '__main__':
    worker()