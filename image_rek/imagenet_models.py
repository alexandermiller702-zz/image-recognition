import numpy as np
import random
import tensorflow.keras as K
from tensorflow import set_random_seed
from tensorflow.keras.applications import (densenet,
                                           inception_v3,
                                           resnet50,
                                           nasnet,
                                           xception)
from data_base import *

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
set_random_seed(SEED)


def test_models(X_train, y_train, X_test, y_test, epoch=5, batch_size=16, v_split=0.2, model='dense'):
    """
    warning requires estimated 15GB CPU and anywhere over 20GB to maybe 50GB+ GPU because the models are all stored IM.
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param epoch:
    :param batch_size:
    :param v_split:
    :return:
    """

    if model == 'dense':
        instance = densenet.DenseNet169(weights='imagenet', input_shape=(300, 300, 3), include_top=False)
    elif model == 'incep_v3':
        instance = inception_v3.InceptionV3(weights='imagenet', input_shape=(300, 300, 3), include_top=False)
    elif model == 'nas':
        instance = nasnet.NASNetMobile(weights='imagenet', input_shape=(300, 300, 3), include_top=False)
    elif model == 'res':
        instance = resnet50.ResNet50(weights='imagenet', input_shape=(300, 300, 3), include_top=False)
    elif model == 'xcept':
        xception.Xception(weights='imagenet', input_shape=(300, 300, 3), include_top=False)
    else:
        print("Error: unsuported model type.")
        return None

    ###################################################
    # # Optional code for Tensorboard call backs.
    # log_dir = './logs/{}'
    # if os.path.exists(log_dir.format(model)):
    #     os.makedirs(log_dir.format(model))
    # tbs = K.callbacks.TensorBoard(log_dir=log_dir.format(model),
    #                                     histogram_freq=0, # 0 == None
    #                                     batch_size=32,
    #                                     write_graph=True,
    #                                     write_grads=False, # only works when hist_fre >0
    #                                     write_images=False)
    ###################################################
    print(model)
    stack = K.models.Sequential()
    stack.add(instance)
    stack.add(K.layers.Flatten())
    stack.add(K.layers.Dense(1024, activation='relu'))
    stack.add(K.layers.Dropout(0.5))
    stack.add(K.layers.Dense(3, activation='sigmoid'))
    ###################################################
    stack.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    ###################################################
    stack.fit(
        X_train, y_train,
        epochs=epoch,
        batch_size=batch_size,
        validation_split=v_split  # , callbacks=[tbs]
    )
    ###################################################
    scores = stack.evaluate(X_test, y_test)
    ###################################################
    return scores


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_train_test_split([20, 10])
    pass
    score = test_models(
        X_train, y_train, X_test, y_test,
        epoch=5, batch_size=16, v_split=0.2,
        model='dense'
    )
    pass
    print(score)



