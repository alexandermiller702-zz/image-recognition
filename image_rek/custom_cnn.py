import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense,
                                     Convolution2D, SeparableConvolution2D,
                                     Dropout,
                                     MaxPooling2D,
                                     GlobalMaxPooling2D,
                                     Flatten)
import tensorflow.keras as K
from keras_preprocessing.image import ImageDataGenerator


class Model:
    """
    A class for handling the behaviour of generic Keras models.
    """
    def __init__(self, image_size=(300, 300, 3), classes=3):
        """
        A class for initialising a consistent architecture for a CNN.
        :param image_size: The WIDTH, HEIGHT, and COLOUR_CHANNELS of your images
        :param classes: The number of distinct classes
        """
        self.img_size = image_size
        self.classes = classes

        classifier = Sequential()
        # Base layer
        classifier.add(
            Convolution2D(
                64, 3, 3,
                input_shape=image_size,
                activation='relu'
            )
        )
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        classifier.add(Convolution2D(128, 3, 3, activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        classifier.add(Convolution2D(256, 3, 3, activation='relu', padding='same'))
        classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        classifier.add(Convolution2D(512, 3, 3, activation='relu', padding='same'))
        classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        classifier.add(Convolution2D(1024, 3, 3, activation='relu', padding='same'))

        classifier.add(Flatten())

        classifier.add(Dense(units=1024, activation='relu'))
        classifier.add(Dense(units=3, activation='sigmoid'))

        self.classifier = classifier
        ########################### Optimiser
        self.optimiser = K.optimizers.Adam()
        ########################### Metrics
        self.mets = [
            K.metrics.categorical_accuracy
            # K.metrics.cosine_proximity,
            # K.metrics.top_k_categorical_accuracy
        ]
        ########################### Loss function
        self.loss = K.losses.categorical_crossentropy  # K.losses.categorical_hinge #
        ########################### Compile
        self.classifier.compile(
            optimizer=self.optimiser,
            loss=self.loss,
            metrics=self.mets
        )

    def callback(self, log_dir='./logs/new', type='csv'):
        """

        :param log_dir:
        :param type: can be CSV or tensorboard.
        :return:
        """
        if os.path.exists(log_dir):
            os.makedirs(log_dir)

        if type == 'tb':
            self.cb = K.callbacks.TensorBoard(
                log_dir='./logs/custom',
                histogram_freq=0,  # 0 == None
                batch_size=32,
                write_graph=True,
                write_grads=False,  # only works when hist_fre >0
                write_images=False
            )
        elif type == 'csv':
            self.cb = K.callbacks.CSVLogger(filename=log_dir+'csv_name.csv')
        else:
            print('unrecgonised callback type.')


    def generate_transformer(self):
        """
        A function for adding an image transformer to the model
        """
        self.transformer= ImageDataGenerator(
            zoom_range=0.2,  # randomly zoom into images
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True  # randomly flip images
        )

    def fit_classifier(self, X_train, Y_train, BATCH_SIZE, EPOCHS, cb=False, transform=False, v=0):
        """
        Generators don't support batch sizes and validation sets.
        :param X_train:
        :param Y_train:
        :param BATCH_SIZE:
        :param EPOCHS:
        :param cb: to use callbacks
        :param transform: To use a image data generator
        :param v: Verbose (binary)
        :return:
        """
        if transform and cb:
            self.callback()
            self.generate_transformer()
            self.classifier.fit_generator(
                self.transformer.flow(X_train, Y_train),
                epochs=EPOCHS,
                verbose=v,
                callbacks=[self.cb]
            )
        elif transform:
            self.generate_transformer()
            self.classifier.fit_generator(
                self.transformer.flow(X_train, Y_train),
                epochs=EPOCHS,
                verbose=v
            )
        elif cb:
            self.callback()
            self.classifier.fit(
                X_train, Y_train,
                epochs=EPOCHS,
                verbose=v,
                batch_size=BATCH_SIZE,
                validation_split=0.2,
                callbacks=[self.cb]
            )
        else:
            self.classifier.fit(
                X_train, Y_train,
                verbose=v,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=0.2
            )

    def evaluate_model(self, X_test, y_test):
        """

        :param X_test:
        :param y_test:
        :return: [Loss, Accuracy]
        """
        score = self.classifier.evaluate(X_test, y_test, verbose=0)
        return score


if __name__ == '__main__':
    pass
