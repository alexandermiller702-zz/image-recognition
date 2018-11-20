import cv2
import os
import random
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from imutils.paths import list_images
from tensorflow.keras.preprocessing.image import img_to_array

SEED = 309
np.random.seed(SEED)
random.seed(SEED)


def get_data_paths(test_data_dir='./data/test'):
    """

    :param test_data_dir:
    :return:
    """
    imagePaths = sorted(list(list_images(test_data_dir)))
    return imagePaths


def load_images(image_paths=None, image_size=(300, 300)):
    """
    Load images from local directory

    :param image_paths:
    :param image_size:
    :return: the image list (encoded as an array)
    """
    # loop over the input images
    if image_paths is None:
        image_paths = get_data_paths()

    images_data = []
    labels = []

    for imagePath in image_paths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (image_size[0], image_size[1]))
        image = img_to_array(image)
        images_data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    return images_data, sorted(labels)


def convert_img_to_array(images, labels):
    """

    :param images:
    :param labels:
    :return: X, y
    """
    # Convert to numpy and do constant normalize
    X = np.array(images, dtype="float") / 255.0
    y = np.array(labels)

    # Binarize the labels
    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    return X, y


def preprocess_data(X):
    """
    Pre-process the test data.
    :param X: the original data
    :return: the preprocess data
    """
    # NOTE: # If you have conducted any pre-processing on the image,
    # please implement this function to apply onto test images.
    return X


def load_train_test_split(t_size=[20,10], data_dir="./data/test/", image_size=(300, 300)):
    """
    A function that produces separated training and testing data.
    :param t_size: [train_size, test_size]
    :param data_dir: data location
    :param image_size: (WIDTH, HEIGHT) of the images
    :return: X_train, y_train, X_test, y_test
    """
    ps_up = sorted(list(list_images(data_dir)))
    ps_up = [ps.replace("\\", "") for ps in ps_up]
    # build a range as pointers for image files
    temp = range(0, len(ps_up))

    # randomly sample for our training set
    train_indices = np.random.choice(temp, size=t_size[0], replace=False)
    train = [ps_up[i] for i in train_indices]

    X_train, y_train = load_images(image_paths=train, image_size=image_size)
    X_train, y_train = convert_img_to_array(X_train, y_train)

    # exclude those training images from our test set
    temp = [i for i in temp if i not in train]
    # subset for testing
    test_indices = np.random.choice(temp, size=t_size[1], replace = False)
    test = [ps_up[i] for i in test_indices]

    X_test, y_test = load_images(image_paths=test, image_size=image_size)
    X_test, y_test = convert_img_to_array(X_test, y_test)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    test_data_dir = "./data/train_data"
    print(load_train_test_split(data_dir=test_data_dir))