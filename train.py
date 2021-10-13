import tables as tb
import random
import tensorflow as tf
import sys
import numpy as np
import datetime
from tensorflow.keras.callbacks import TensorBoard
from models import BasicCNN
from dataGenerators import DataGenerator

batch_size = 30
nr_channels = 5

file = tb.open_file('./data/preprocessed/BTCUSDT_5m_data.h5', 'r')

data_size = file.root.data.close.shape
data_length = data_size[0]
img_size = data_size[1]


# Train Validation Split
split = 0.8
idx = list(np.arange(0, data_length))
training_idx = idx[:int(split*len(idx))]
validation_idx = idx[int(split*len(idx)):]


train_generator = DataGenerator(file, training_idx, batch_size)
val_generator = DataGenerator(file, validation_idx, batch_size)

# print(tf.size(train_generator))
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model = BasicCNN(img_size, batch_size, nr_channels)

model.fit_generator(generator = train_generator,
                    validation_data = val_generator,
                    epochs=15,
                    callbacks=[tensorboard_callback])

model.save('CNN-T2-3M5m.h5')