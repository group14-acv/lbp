from os import listdir

ELEPHANT_SKIN = "imgs/elephant_skin.jpg"
HUMAN_SKIN = "imgs/human_skin.jpg"
TEST = "./imgs/test.jpg"
CARPET_TRAIN = ["imgs/carpet_training/img1.png", "imgs/carpet_training/img2.png", "imgs/carpet_training/img3.png", "imgs/carpet_training/img4.png"]
CARPET_TEST = ["imgs/carpet/img1.png", "imgs/carpet/img2.png", "imgs/carpet/img3.png", "imgs/carpet/img4.png"]
PICKLE_FILE = "svm_model.pkl"


# get functions used for retrieving filenames
def get_training_imgs():
    res = []
    for dirs in listdir("./imgs/training"):
        res.extend(['./imgs/training/' + dirs + "/" + filename for filename in listdir("./imgs/training/" + dirs)])
    return res

def get_training_test_imgs():
    res = []
    for dirs in listdir("./imgs/training"):
        res.extend(['./imgs/training/' + dirs + "/" + filename for filename in listdir("./imgs/training/" + dirs)])
    return res[0:2] + res[-3:-1]

def get_testing_imgs():
    return ['./imgs/testing/' + x for x in listdir("./imgs/testing")]

# pickle functions used for saving and loading support vector machine models to save time
def pickle_save(model):
    import pickle

    output = open(PICKLE_FILE, 'wb')
    pickle.dump(model, output)


def pickle_load():
    import pickle

    pkl_file = open(PICKLE_FILE, 'rb')
    return pickle.load(pkl_file)
