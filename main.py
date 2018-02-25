import numpy as np
import cv2
import consts
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

test_matrix = np.array([[1, 4, 6], [9, 7, 1], [5, 7, 9]])

test_matrix2 = np.array([[5, 4, 2, 2, 1], [3, 5, 8, 1, 3], [2, 5, 4, 1, 2], [4, 3, 7, 2, 7], [1, 4, 4, 2, 6]])

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
# res = start_3_by_3(imgs.TEST)
#
# data = []
# labels = []
#
# for img_path in consts.get_training_imgs():
#     img_res = start_3_by_3(img_path)
#
#     labels.append(img_path.split('/')[-2])
#
#     (hist, _) = np.histogram(img_res, np.arange(0, 25))
#     hist = hist / np.sum(hist)
#     data.append(hist)
#
# model = LinearSVC(C=100.0)
# model.fit(data, labels)
#
# consts.pickle_save(model)

model = consts.pickle_load()

for img_path in consts.get_testing_imgs():
    img_res = start_3_by_3(img_path)

    (hist, _) = np.histogram(img_res, np.arange(0, 25))
    hist = hist / np.sum(hist)
    print(hist)

    hist = hist.reshape(1, -1)

    prediction = model.predict(hist)
    prediction = prediction[0]

    img = cv2.imread(img_path)

    # display the image and the prediction
    cv2.putText(img, prediction, (10, 30), cv2.FONT_ITALIC,
                1.0, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
