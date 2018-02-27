import numpy as np
import cv2
import consts
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# traditional local binary pattern using 3x3 matrix
def start_3_by_3(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape

    new_height = height - 2
    new_width = width - 2
    print("filename:", filename)
    print("height:", height, "| width:", width)
    print("new_height:", new_height, "| new_width:", new_width)

    new_matrix = np.zeros(shape=(new_width, new_height))

    for y in range(new_height):
        y += 1
        for x in range(new_width):
            x += 1
            center = (y, x)
            indexes = get_subwindow_indexes(center)
            res = int(compare_neighbours(center, img, indexes), 2)
            new_matrix[x - 1, y - 1] = res

    return new_matrix


def compare_neighbours(center, image, indexes):
    # get center pixel
    pixel = image.item(center)

    res = ""

    #  print(indexes)
    for index in indexes:
        # get neighbour pixel
        n_pixel = image.item(index)

        if (pixel >= n_pixel):
            res += "1"
        else:
            res += "0"

    return res

def get_subwindow_indexes(center):
    indexes = get_neighbours_indexes(clockwise=False)
    x, y = center

    return [(a_x + x - 1, a_y + y - 1) for (a_x, a_y) in indexes]


def get_neighbours_indexes(clockwise=True, scale=1):
    if (clockwise and scale == 1):
        return [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)]
    elif (not clockwise and scale == 1):
        return [(0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]


# def get_mask(a, b, n, r=1):
#     y, x = np.ogrid[-a:n - a, -b:n - b]
#     mask = x * x + y * y <= r * roo
#

res = start_3_by_3(consts.TEST)
res_array = np.asarray(res).reshape(-1)

import csv
with open('shinigami.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(res_array)
