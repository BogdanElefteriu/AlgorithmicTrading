import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter
from pyts.image import RecurrencePlot

class ImageBuilder():

    def __init__(self, data):
        self.data = data

    def dist_mat(self, im_size, time, sigma):
        images = np.empty(shape=(time - im_size + 1, im_size, im_size))
        delim = im_size
        for i in range(0, time-im_size + 1):
            dist_mat = euclidean_distances(self.data[i:delim].values.reshape(-1,1))
            delim += 1
            images[i,:,:] = gaussian_filter(dist_mat,sigma = sigma)
        return images

    def normalise(self, images, im_size = 200, time = 281):
        norm_images = np.empty(shape=((time - im_size) + 1, im_size, im_size))
        for i in range(0, images.shape[0]):
            norm_images[i,:,:] = normalize(images[i,:,:], norm = 'max')
        return norm_images

    def recurrence_plot(self,im_size, time, threshold = 20):
        images = np.empty(shape=(time - im_size + 1, im_size, im_size))
        rp = RecurrencePlot(threshold='point', percentage=threshold)
        delim = im_size
        for i in range(0, time - im_size + 1):
            X_rp = rp.fit_transform(self.data[i:delim].values.reshape(-1,1).transpose())
            delim+=1
            images[i,:,:] = X_rp
        return images