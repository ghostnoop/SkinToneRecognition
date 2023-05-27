import io
import os
import pickle
from typing import List

import face_recognition
import pandas as pd
from face_recognition import face_locations

from skin_module.service import find_nearest_coordinates


class Prediction:
    model_name = 'face_model.pkl'
    COLS = ['White']
    N_UPSCLAE = 1
    model_path = os.path.join(os.path.dirname(__file__), model_name)

    def __init__(self):
        with open(Prediction.model_path, 'rb') as f:
            clf, labels = pickle.load(f, encoding='latin1')
        self.clf = clf
        self.labels = labels

    def predict(self, img: bytes, coords: List[dict]):
        img = io.BytesIO(img)
        try:
            pred, locs = Prediction.predict_one_image(img, self.clf, self.labels)
            if not locs:
                print('no locs')
                return {"message":None}
        except Exception as e:
            print(e)
            print("Skipping")

            return {"message":None}

        return Prediction.mapping_results_with_entered_coords(Prediction.format_results(pred, locs), coords)

    @staticmethod
    def format_results(pred, locs):
        coords_label = ['top', 'right', 'bottom', 'left']
        locs = \
            pd.DataFrame(locs, columns=coords_label)
        # print(locs)
        df = pd.concat([pred, locs], axis=1)

        coords_to_white = []

        for index, row in df.iterrows():
            t = {}
            for label in coords_label:
                t[label] = row[label]
            coords_to_white.append({'key': t, 'value': row['White']})
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

    @staticmethod
    def extract_features(img_path):
        """Exctract 128 dimensional features"""
        X_img = face_recognition.load_image_file(img_path)
        locs = face_locations(X_img, number_of_times_to_upsample=Prediction.N_UPSCLAE, model="rcnn")
        if len(locs) == 0:
            return None, None
        face_encodings = face_recognition.face_encodings(X_img, known_face_locations=locs)
        return face_encodings, locs

    @staticmethod
    def predict_one_image(img_path, clf, labels):
        """Predict face attributes for all detected faces in one image"""
        face_encodings, locs = Prediction.extract_features(img_path)
        if not face_encodings:
            return None, None

        clf_result = clf.predict_proba(face_encodings)
        pred = pd.DataFrame(clf_result,
                            columns=labels)

        pred = pred.loc[:, labels]
        return pred, locs


if __name__ == '__main__':
    with open('post_3.jpg', 'rb') as f:
        data = f.read()

    a = [{'left': 232, 'top': 152, 'right': 342, 'bottom': 286}, {'left': 413, 'top': 72, 'right': 530, 'bottom': 228},
         {'left': 660, 'top': 217, 'right': 718, 'bottom': 292}, {'left': 405, 'top': 198, 'right': 438, 'bottom': 249}]

    p = Prediction()
    results_ = p.predict(data, a)
    print(results_)
