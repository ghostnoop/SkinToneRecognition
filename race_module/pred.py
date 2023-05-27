from __future__ import print_function

import io
import pickle

import cv2
import face_recognition
import numpy as np
import pandas as pd
from face_recognition import face_locations

# we are only going to use 4 attributes
COLS = ['White']
N_UPSCLAE = 1


def extract_features(img_path):
    """Exctract 128 dimensional features
    """
    X_img = face_recognition.load_image_file(img_path)
    locs = face_locations(X_img, number_of_times_to_upsample=N_UPSCLAE, model="rcnn")
    if len(locs) == 0:
        return None, None
    face_encodings = face_recognition.face_encodings(X_img, known_face_locations=locs)
    return face_encodings, locs


def predict_one_image(img_path, clf, labels):
    """Predict face attributes for all detected faces in one image
    """
    face_encodings, locs = extract_features(img_path)
    if not face_encodings:
        return None, None

    clf_result = clf.predict_proba(face_encodings)
    pred = pd.DataFrame(clf_result,
                        columns=labels)

    pred = pred.loc[:, labels]
    return pred, locs


def main():
    model_path = 'face_model.pkl'

    with open(model_path, 'rb') as f:
        clf, labels = pickle.load(f, encoding='latin1')
    # for i in [f'post_{i}.jpg' for i in range(5)]:
    for i in ['aaa.jpg']:
        with open(i, 'rb') as f:
            img = f.read()
        # img = 'img.png'
        img = io.BytesIO(img)
        try:
            pred, locs = predict_one_image(img, clf, labels)
            if not locs:
                print('no locs')
                continue
        except Exception as e:
            print(e)
            print("Skipping {}")
            return
        # print(pred)
        # print(locs)
        coords_label = ['top', 'right', 'bottom', 'left']
        locs = \
            pd.DataFrame(locs, columns=coords_label)
        # print(locs)
        df = pd.concat([pred, locs], axis=1)

        coords_to_white = {}

        for index, row in df.iterrows():
            t = {}
            for label in coords_label:
                t[label] = row[label]
            coords_to_white[t] = row['White']

        print(i)
        print(df)
    # img = draw_attributes(img, df)


a = [{'left': 232, 'top': 152, 'right': 342, 'bottom': 286},
     {'left': 413, 'top': 72, 'right': 530, 'bottom': 228},
     {'left': 660, 'top': 217, 'right': 718, 'bottom': 292},
     {'left': 405, 'top': 198, 'right': 438, 'bottom': 249}]

if __name__ == "__main__":
    main()
