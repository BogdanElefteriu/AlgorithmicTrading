import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    def __init__(self, data_file, indices, batch_size, shuffle=False):
        self.data_file = data_file
        self.batch_size = batch_size
        self.indices = indices
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        # Generates one batch of data
        images = []
        labels = []

        lwr = index * self.batch_size
        upr = ((index + 1) * self.batch_size)

        labels.append(self.data_file.root.label[:, self.indices[lwr:upr]])
        labels = np.squeeze(np.array(labels))
        labels = to_categorical(labels, 3)

        for im in self.data_file.root.data:
            images.append(np.array(im[self.indices[lwr:upr], :, :]))
        images = np.moveaxis(np.array(images), 0, -1)

        # print('images : shape = %s, type = %s' % (images.shape, images.dtype))
        # print('labels : shape = %s, type = %s' % (labels.shape, labels.dtype))

        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
