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
        # color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color_img = img
        # try:
        # st = time.monotonic()
        try:
            return DeepFace.analyze(color_img, actions=('race',), align=False, silent=True, detector_backend='opencv')
        except Exception as e:
            print(e)
            return DeepFace.analyze(color_img, actions=('race',), align=False, silent=True,
                                    detector_backend='retinaface')

    def __predict(self, img):
        # cv2.imshow("cropped", img)
        # cv2.waitKey(0)
        try:
            return DeepFace.analyze(img, actions=('race',), align=False, silent=True, detector_backend='opencv')
        except Exception as e:
            print(e)
            return DeepFace.analyze(img, actions=('race',), align=False, silent=True,
                                    detector_backend='retinaface')

    def predict(self, img: bytes, coords: List[dict]):
        prediction = self._predict(img)
        return Prediction.mapping_results_with_entered_coords(Prediction.format_results(prediction), coords)

    def predict_with_crop(self, img: bytes, coords: List[dict]):
        coords_to_white = []
        nparr = np.fromstring(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for coord in coords:
            x, y, x1, y1 = list(coord.values())
            N=30
            # crop_img = img[y-N:y1+N, x-N:x1+N]
            crop_img = img
            prediction = self.__predict(crop_img)
            coords_to_white.append(
                {'key': coord, 'value': Prediction.format_results_1(prediction)}
            )

        return coords_to_white

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
    def format_results_1(prediction: dict):
        for face in prediction:
            race = face["dominant_race"]
            race_value = face["race"][race]
            if race in ('white', 'middle eastern', 'latino hispanic') and race_value >= 35:
                return race_value if face["race"][race] >= 60 else 60
            return 0

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
