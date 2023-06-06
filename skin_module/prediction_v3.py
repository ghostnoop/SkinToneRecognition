from typing import List

import cv2
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor


def patch_asscalar(a):
    return np.asarray(a).item()


setattr(np, "asscalar", patch_asscalar)

default_categories = ["#373028", "#422811", "#513b2e", "#6f503c",
                      "#81654f", "#9d7a54", "#bea07e", "#e5c8a6",
                      "#e7c1b8", "#f3dad6", "#fbf2f3"]
category_to_color = {'#373028': 'black', '#422811': 'black', '#513b2e': 'black',
                     '#6f503c': 'black', '#81654f': 'white', '#9d7a54': 'white',
                     '#bea07e': 'white', '#e5c8a6': 'white', '#e7c1b8': 'white',
                     '#f3dad6': 'white', '#fbf2f3': 'white'}

default_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(default_categories)]
debug: bool = False
categories: list[str] = default_categories
cate_labels = default_labels
for idx, ct in enumerate(categories):
    if not ct.startswith('#') and len(ct.split(',')) == 3:
        r, g, b = ct.split(',')
        categories[idx] = '#%02X%02X%02X' % (int(r), int(g), int(b))
n_dominant_colors = 2


class Prediction:
    def detect_skin(self, image):
        img = image.copy()
        # Converting from BGR Colours Space to HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Defining skin Thresholds
        low_hsv = np.array([0, 48, 80], dtype=np.uint8)
        high_hsv = np.array([20, 255, 255], dtype=np.uint8)

        skinMask = cv2.inRange(img, low_hsv, high_hsv)
        skinMask = cv2.GaussianBlur(skinMask, ksize=(3, 3), sigmaX=0)
        skin = cv2.bitwise_and(img, img, mask=skinMask)

        all_0 = np.isclose(skin, 0).all()
        return image if all_0 else cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

    def create_bar(self, height, width, color):
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = color
        return bar

    def dominant_colors(self, image, n_clusters=3):
        data = np.reshape(image, (-1, 3))
        data = data[np.all(data != 0, axis=1)]
        data = np.float32(data)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, colors = cv2.kmeans(data, n_clusters, None, criteria, 10, flags)
        labels, counts = np.unique(labels, return_counts=True)

        order = (-counts).argsort()
        colors = colors[order]
        counts = counts[order]

        props = counts / counts.sum()

        return colors, props

    def skin_label(self, colors, props, categories, cate_labels):
        lab_labels = [convert_color(sRGBColor.new_from_rgb_hex(lbl), LabColor) for lbl in categories]
        lab_colors = [convert_color(sRGBColor(rgb_r=r, rgb_g=g, rgb_b=b, is_upscaled=True), LabColor) for b, g, r in
                      colors]
        distances = [np.sum([delta_e_cie2000(c, label) * p for c, p in zip(lab_colors, props)]) for label in lab_labels]
        label_id = np.argmin(distances)
        distance: float = distances[label_id]
        category_hex = categories[label_id]
        PERLA = cate_labels[label_id]
        return label_id, category_hex, PERLA, distance

    def classify(self, image, n_dominant_colors, categories, cate_labels):
        image = image.copy()
        image = self.detect_skin(image)

        colors, props = self.dominant_colors(image, n_clusters=n_dominant_colors)

        # Generate readable strings
        hex_colors = ['#%02X%02X%02X' % tuple(np.around([r, g, b]).astype(int)) for b, g, r in colors]
        prop_strs = ['%.2f' % p for p in props]
        res = list(np.hstack(list(zip(hex_colors, prop_strs))))
        label_id, category_hex, PERLA, distance = self.skin_label(colors, props, categories, cate_labels)
        distance = round(distance, 2)
        res.extend([category_hex, PERLA, distance])

        debug_img = None

        return res, debug_img

    def predict(self, image: bytes, coords: List[dict]):
        nparr = np.fromstring(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        coords_to_white = []

        for coord in coords:
            x1, y1, x2, y2 = list(coord.values())
            face = img[y1:y2, x1:x2]
            res, _debug_img = self.classify(face, n_dominant_colors, categories, cate_labels)
            color = category_to_color[res[4]]
            print(color,res[4])
            coords_to_white.append({'key': coord, 'value': 100 if color == 'white' else 0})
        return coords_to_white
