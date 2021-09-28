import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ConvLSTM2D, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence


# Divide data in training and testing sets

labels = to_categorical(data.label[imageSize-1:], num_classes=3, dtype='float64')  # Col1(0) = Hold // Col2(1) = Buy // Col3(2) = Sell

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(allImages, labels, test_size= 0.25)


## Convolutional Neural Network

# NAME = "CNN-TEST1-3M5m"

# !rm -rf ./logs/
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback=TensorBoard(log_dir=log_dir, histogram_freq=1)
# tensorboard = TensorBoard(log_dir= 'logs/{}'.format(NAME))

model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = allImages.shape[1:]))
model.add(Activation("relu"))
model.add(Dropout(rate=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3), input_shape = allImages.shape[1:]))
model.add(Activation("relu"))
model.add(Dropout(rate=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3), input_shape = allImages.shape[1:]))
model.add(Activation("relu"))
model.add(Dropout(rate=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = allImages.shape[1:]))
model.add(Activation("relu"))
model.add(Dropout(rate=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dropout(rate=0.2))

model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss=CategoricalCrossentropy(),
                optimizer="adam",
                metrics=['accuracy'])

model.fit(train_X,train_Y, epochs=15, batch_size=36, validation_data=(test_X, test_Y), callbacks=[tensorboard_callback])  # validation_split=0.25) #

model.save('CNN-T2-3M5m.h5')
