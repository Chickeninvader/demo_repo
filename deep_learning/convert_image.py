"""
This program is currently combine processing image and training
"""
import glob
import cv2
import numpy as np
from keras.preprocessing.image import array_to_img
from brain import brain
from forward_propagation import forward
from check_accuracy import check_result
# from forward_propagation import forward

cat_link = "D:/anhbin/programming_python/deep_learning/dogvscat/train/cat/*.jpg" 
dog_link = "D:/anhbin/programming_python/deep_learning/dogvscat/train/dog/*.jpg"
test_cat = "D:/anhbin/programming_python/deep_learning/dogvscat/test/cat/*.jpg"
test_dog = "D:/anhbin/programming_python/deep_learning/dogvscat/test/dog/*.jpg"


def convert_image(*directory):
    """
    Argument: 
    directory -- list of directory where we want to convert picture (in this program, we limit the use to
    only 2 file contains training and test set of cats and dog)

    Return:
    image1 -- 2D array contain detail of pictures (size_of_picture*3, number_of_examples)
    outcome -- an array contain information of picture (1, number_of_examples)
    We choose 0 = cat, 1 = dog
    """
    image1 = []
    outcome = []
    for dir in directory:
        for file in glob.glob(dir):
            image = cv2.imread(file)
            # We resize every picture to 100*100 and divide all elements so that all of them is less than 1
            image = cv2.resize(image, (100,100), interpolation=cv2.INTER_AREA)/255.
            image1.append(image)
            if dir == "D:/anhbin/programming_python/deep_learning/dogvscat/train/cat/*.jpg" or dir == "D:/anhbin/programming_python/deep_learning/dogvscat/test/cat/*.jpg":
                outcome.append(0)
            else:
                outcome.append(1)

    outcome = np.array(outcome)
    outcome = outcome.reshape(1, outcome.shape[0])
    image1 = np.array(image1)
    image1 = np.reshape(image1, (image1.shape[0], -1))
    image1 = np.transpose(image1)

    return image1, outcome


def show_image(index, array_of_image):
    image = np.reshape(array_of_image[:, index], (100,100,3))*255
    image = array_to_img(image)
    image.show()

# Training
dog_cat_train, outcome = convert_image(cat_link, dog_link)
parameters = brain(X=dog_cat_train, Y=outcome, iteration=20, learning_rate=5)

# Predict
dog_cat_test, outcome = convert_image(test_cat, test_dog)
A2, cache = forward(X= dog_cat_test, parameters=parameters)
print(check_result(A2, Y=outcome))


