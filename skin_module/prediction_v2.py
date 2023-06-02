import time
from typing import List

import cv2
import numpy as np
from deepface import DeepFace

from skin_module.service import find_nearest_coordinates


class Prediction:
    def __init__(self):
        pass

    def _predict(self, img: bytes):
        nparr = np.fromstring(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # try:
        # st = time.monotonic()
        try:
            return DeepFace.analyze(color_img,actions=('race',),align=False,silent=True, detector_backend='opencv')
        except Exception as e:
            print(e)
            return DeepFace.analyze(color_img,actions=('race',),align=False,silent=True, detector_backend='retinaface')
        # print(time.monotonic()-st)


        # except:
        #     print('eeee')
        #     return DeepFace.analyze(color_img, silent=True, detector_backend='retinaface')

    def predict(self, img: bytes, coords: List[dict]):
        prediction = self._predict(img)
        return Prediction.mapping_results_with_entered_coords(Prediction.format_results(prediction), coords)

    @staticmethod
    def format_results(prediction: dict):
        coords_to_white = []

        for face in prediction:
            t = {}
            race = face["dominant_race"]
            k = face['region']
            coords = dict(left=k['x'], top=k['y'], right=k['x'] + k['w'], bottom=k['y'] + k['h'])
            t['key'] = coords
            race_value = face["race"][race]
            if race in ('white', 'middle eastern', 'latino hispanic') and race_value >= 35:
                t['value'] = race_value if face["race"][race] >= 60 else 60
                print(face["race"][race])
            else:
                print(race, face["race"][race])
                t['value'] = 0
            coords_to_white.append(t)

        return coords_to_white

    @staticmethod
    def mapping_results_with_entered_coords(coords_to_white: dict, coords: List[dict]):
        keys = [i['key'] for i in coords_to_white]

        def _find(_coord):
            for index, i in enumerate(coords_to_white):
                if i['key'] == _coord:
                    return i['value'], index
            return None, None

        results = []
        for coord in coords:
            nearest_coords = find_nearest_coordinates(keys, coord)
            value, index_ = _find(nearest_coords)
            results.append({'key': coord, 'value': value})
            if index_:
                del keys[index_]
                del coords_to_white[index_]

        return results
