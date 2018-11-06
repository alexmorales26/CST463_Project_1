# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

cifar_file ="data_batch_1"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cf = unpickle(cifar_file)

# see what's in the dictionary

cf.keys()
cf[b'filenames']
cf[b'batch_label']

# get the data and labels

dat = cf[b'data']
labels = cf[b'labels']

# reshape image data and display a grid of images
# from https://stackoverflow.com/questions/35995999

m = 32
n = 32
X = dat.reshape(10000, 3, m, n).transpose(0,2,3,1).astype("uint8")
y = np.array(labels)

nrows = 10
ncols = 10
fig, axes1 = plt.subplots(nrows,ncols,figsize=(8,8))
for j in range(nrows):
    for k in range(ncols):
        i = np.random.choice(range(len(X)))
        img = X[i:i+1][0]
        axes1[j][k].set_axis_off()
        axes1[j][k].imshow(img)