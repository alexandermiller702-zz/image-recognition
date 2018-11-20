
#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""

import tensorflow.keras as K
import numpy as np
import os
import tensorflow as tf
import random

from data_base import *
from custom_cnn import *

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)


def construct_model(img_size=(300, 300 ,3), classes=3):
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    return Model(image_size=img_size, classes=classes)


def train_model(model, X_train, y_train, BATCH_SIZE, EPOCHS, tb=False, transform=False):
    """

    :param model:
    :param X_train:
    :param y_train:
    :param BATCH_SIZE:
    :param EPOCHS:
    :param tb:
    :param transform:
    :return:
    """
    # Add your code here
    return model.fit_classifier(X_train, y_train,
                                BATCH_SIZE, EPOCHS,
                                cb=tb, transform=transform)


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    # model.save("model/model.h5")
    print("Model Saved Successfully.")


if __name__ == '__main__':
    t_size = [2500, 200]
    BATCH_SIZE = 128
    CLASSES = 3
    EPOCHS = 15
    image_size = (300, 300, 3)
    test_data_dir = "./data/train_data"
    ###########################
    X_train, y_train, X_test, y_test = load_train_test_split(t_size, test_data_dir, image_size)
    ###########################
    model = construct_model(image_size, CLASSES)
    model.fit_classifier(X_train, y_train, BATCH_SIZE, EPOCHS, cb=True, transform=False, v=1)
    ###########################
    from test import independent_test_data
    X_test, y_test, _index = independent_test_data((image_size[0], image_size[1]))
    score = model.evaluate_model(X_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    ###########################
    from pandas import DataFrame, concat
    a = DataFrame(model.classifier.predict(X_test))
    b = DataFrame(y_test)
    a['paths'] = [i.split('/')[4] for i in _index]
    a= concat([a, b], axis=1)
    a.to_csv("thirds.csv")
    print(a)

    # save_model(model)
