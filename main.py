
#!/usr/bin/env python

"""Description:
The main.py is to build your CNN model, train the model, and save it for later evaluation(marking)
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

from image_rek.data_base import *
from image_rek.custom_cnn import *

# Set random seeds to ensure the reproducible results
SEED = 501
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)


BATCH_SIZE = 128
CLASSES = 3
EPOCHS = 15


t_size = [2500, 200]
image_size = (300, 300, 3)
test_data_dir = "./data/train_data"


if __name__ == '__main__':
    ###########################
    X_train, y_train, X_test, y_test = load_train_test_split(t_size, test_data_dir, image_size)
    ###########################
    model = Model(image_size=image_size, classes=CLASSES)
    model.fit_classifier(
        X_train, y_train,
        BATCH_SIZE, EPOCHS,
        cb=True,
        transform=False,
        v=1
    )
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
    ###########################
    model.save("model/model.h5")
