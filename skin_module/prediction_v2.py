from typing import List

import cv2
import numpy as np
from deepface import DeepFace

from skin_module.service import find_nearest_coordinates


class Prediction:
    def __init__(self):
        pass

    def predict(self, img: bytes, coords: List[dict]):
        nparr = np.fromstring(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prediction = DeepFace.analyze(color_img)
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

            if race == 'white' and face["race"][race] >= 50:
                t['value'] = face["race"][race]
            else:
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