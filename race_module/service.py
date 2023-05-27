import math


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_nearest_coordinates(a, b):
    min_distance = float('inf')
    nearest_coordinates = None

    for coord in a:
        distance = euclidean_distance(coord['left'], coord['top'], b['left'], b['top'])
        if distance < min_distance and distance<30:
            min_distance = distance
            nearest_coordinates = coord

    return nearest_coordinates

# nearest_coords = find_nearest_coordinates(a, b)
