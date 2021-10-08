import numpy as np
import tables
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, data_file, indices, batch_size, shuffle=True):
        self.data_file = data_file
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        # Generates one batch of data
        X = []
        y = []
        for i in self.indices[index * self.batch_size: (index + 1) * self.batch_size]:
            X.append(self.data_file.root.data.close[i])
            y.append(self.data_file.root.labels[i])

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
