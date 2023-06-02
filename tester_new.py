import time

import requests

from skin_module.prediction_v2 import Prediction


def worker():
    prediction = Prediction()

    content = requests.get('https://www.tprteaching.com/wp-content/uploads/2022/12/peoples-or-peoples.jpg').content
    st = time.monotonic()
    result = prediction._predict(content)
    print(time.monotonic()-st)

    st = time.monotonic()
    result = prediction._predict(content)
    print(time.monotonic() - st)
    # print(result)

if __name__ == '__main__':
    worker()