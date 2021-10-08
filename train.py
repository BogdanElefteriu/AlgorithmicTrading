import tables
import random
import sys
import numpy as np
import datetime
from tensorflow.keras.callbacks import TensorBoard
from models import BasicCNN
from dataGenerators import DataGenerator

img_size = 200
batch_size = 5
nr_channels = 1

data = tables.open_file('data/preprocessed/BTCUSDT_5m_data.h5')

data_length = len(data.root.data.close)

# Train Validation Split
split = 0.8
idx = random.sample(range(data_length), data_length)
training_idx = idx[:int(split*len(idx))]
validation_idx = idx[int(split*len(idx)):]

train_generator = DataGenerator(data, training_idx, batch_size)
val_generator = DataGenerator(data, validation_idx, batch_size)
X = []
index = 10
for i in idx[index * batch_size: (index + 1) * batch_size]:
    X.append(data.root.data.close[i])
X = np.array(X)
print(data.root.data.length)
sys.exit()
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model = BasicCNN(img_size, batch_size, nr_channels)

model.fit_generator(train_generator, val_generator, epochs=15,
                    callbacks=[tensorboard_callback])

model.save('CNN-T2-3M5m.h5')