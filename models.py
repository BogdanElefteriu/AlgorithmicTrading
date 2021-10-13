from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ConvLSTM2D, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence

def BasicCNN(img_size, batch_size, nr_channels):

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(img_size, img_size, nr_channels)))
    model.add(Activation("relu"))
    model.add(Dropout(rate=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(Dropout(rate=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(Dropout(rate=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(Dropout(rate=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dropout(rate=0.2))

    model.add(Dense(3))
    model.add(Activation('sigmoid'))

    model.compile(loss=CategoricalCrossentropy(),
                  optimizer="adam",
                  metrics=['accuracy'])
    return model
