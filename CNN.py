import tensorflow as tf
import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ConvLSTM2D, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence
from sklearn.model_selection import train_test_split


def CNNLSTM_architecture(model, images):
    """
    Function defining the CNN-LSTM architecture
    :param model: model to be trained
    :param images: training dataset
    """

    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=images.shape[1:]))
    model.add(Activation("relu"))
    model.add(Dropout(rate=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(rate=0.2))

    model.add(Dense(3))
    model.add(Activation('sigmoid'))


def generate_model(images, labels):

    # Divide data in training and testing sets
    labels = to_categorical(labels, num_classes=3, dtype='float32')  # Col1(0) = Hold // Col2(1) = Buy // Col3(2) = Sell
    train_X, test_X, train_Y, test_Y = train_test_split(images, labels, test_size=0.25)

    # Initialise TensorBoard
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = Sequential()
    CNNLSTM_architecture(model, images)
    model.compile(loss=CategoricalCrossentropy(),
                  optimizer="adam",
                  metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=15, batch_size=36, validation_data=(test_X, test_Y), callbacks=[tensorboard_callback])
    model.save('./data/model/tests/CNN-T2-3M5m.h5')
