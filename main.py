from labelling import Label

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


##################################### LABEL GENERATION & TESTING ######################################

## Generate labels

label = Label(data[['close']])
labels = label.generate(window_size = 500)
data = pd.merge_asof(left=data, right=labels[['label']],
                     right_index=True, left_index=True, direction='nearest', tolerance=pd.Timedelta('1 second'))

data.dropna(inplace=True)





###################################### PLOTTING ######################################

## Plot labels on data
# fig, ax = plt.subplots()
#
# ax.plot(data.close.iloc[1:200])
# for i, label in enumerate(data.label):
#     timestamp = data['label'].index[i]
#     labels = str(int(label))
#     x = data.close
#     ax.annotate(labels, xy= (timestamp, x.iloc[i]))
# plt.show()

# a = ax[1,0].imshow(images[:,:,0])
# plt.colorbar(a)


## Plot animation of price and RP transform
# fig, ax = plt.subplots(2)
# # i=520
# # # plt.imshow(normImages1[i,:,:])
# # ax[1].imshow(normImages1[i,:,:])
# # plt.plot(data.Close[i:i+400])
# # plt.show()
# def animate(i):
#     ax[0].plot(data.close.iloc[1:imageSize+i], 'b')
#     print(data.Labels[imageSize + i])
#     ax[1].imshow(normImages1[i,:,:])
#
# animate = FuncAnimation(plt.gcf(), animate, frames = 400, interval=3000)
# plt.tight_layout()
# plt.show()


## Plot indicators
# plt.figure(figsize=(20, 10))
# plt.plot(data['RSI'], color='k', alpha=0.8)
# plt.show()
#
# plt.figure(figsize=(20, 10))
# plt.plot(data['MACD'], color='r', alpha=0.8)
# plt.plot(data['MACDS'], color='b', alpha=0.8)
# plt.show()
#
# plt.figure(figsize=(20, 10))
# plt.plot(data['K'], color='g', alpha=0.8)
# plt.show()