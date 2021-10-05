from preprocess import preprocess
from CNN import generate_model

from tables import *
import h5py
import tftables
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### Preprocess data
data = preprocess()

# file = open_file('./data/preprocessed/BTCUSDT_5m_data.h5', mode = 'r')
# data = file.root.data
#
#
# RSI =data.RSI[:]






# def print_attrs(name, obj):
#     print(name)
#     for key, val in obj.attrs.items():
#         print("    %s: %s" % (key, val))
#
#
# with h5py.File('./data/preprocessed/BTCUSDT_5m_data.h5', 'r') as hdf:
#     group = hdf.get('data')
#     data_items = list(group.items())
#     close = np.array(group.get('close'))
#     print(close.shape)
#     # print(group)
#     # print(data)
#     # print(hdf.visititems(print_attrs))
#     print(names)

#
# print(list(file.keys()))
# Open the HDF5 file and create a loader for a dataset.
# The batch_size defines the length (in the outer dimension)
# of the elements (batches) returned by the reader.
# Takes a function as input that pre-processes the data.
# loader = tftables.load_dataset(filename='./data/preprocessed/BTCUSDT_5m_data.h5',
#                                dataset_path='/data',
#                                batch_size=20)

# # To get the data, we dequeue it from the loader.
# # Tensorflow tensors are returned in the same order as input_transformation
# truth_batch, data_batch = loader.dequeue()
#
# # The placeholder can then be used in your network
# result = my_network(truth_batch, data_batch)
#
#
# print(data.path)
# array_batch_placeholder = file.get_batch(
#     path = '/h5/path',  # This is the path to your array inside the HDF5 file.
#     cyclic = True,      # In cyclic access, when the reader gets to the end of the
#                         # array, it will wrap back to the beginning and continue.
#     ordered = False     # The reader will not require the rows of the array to be
#                         # returned in the same order as on disk.

# a = []
# images = np.empty((752, 200, 200, 5))
# labels = []
#
# for node in data.__iter__():
#     i = 0
#     if node.name == 'label':
#         labels = np.stack(node[:])
#     else:
#         a[i] = node[:]
#         np.append(images[..., i], node[:])
#     i += 1