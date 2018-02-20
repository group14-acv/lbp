ELEPHANT_SKIN = "imgs/elephant_skin.jpg"
HUMAN_SKIN = "imgs/human_skin.jpg"
TEST = "imgs/test.png"
CARPET_TRAIN = ["imgs/carpet_training/img1.png", "imgs/carpet_training/img2.png", "imgs/carpet_training/img3.png", "imgs/carpet_training/img4.png"]
CARPET_TEST = ["imgs/carpet_testing/img1.png", "imgs/carpet_testing/img2.png", "imgs/carpet_testing/img3.png", "imgs/carpet_testing/img4.png"]

from os import listdir

def get_training_imgs():
    res = []
    for dirs in listdir("./imgs/training"):
        res.extend([dirs + "/" + filename for filename in listdir("./imgs/training/" + dirs)])
    return res

def get_testing_imgs():
    return listdir("./imgs/testing")
