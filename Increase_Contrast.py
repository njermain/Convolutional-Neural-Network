# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:40:45 2019

@author: w10007346
"""


from skimage import data, img_as_float
from skimage import exposure
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import os 
os.getcwd()

img=mpimg.imread('C:/Users/w10007346/Dropbox/CNN/AugmentedImages/_2239_1298378.png')
imgplot=plt.imshow(img)

p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
plt.imshow(img_rescale)

# apply to a whole folder now
os.chdir('C:/Users/w10007346/Dropbox/CNN/Contrast')
from os import listdir
from os.path import isfile, join
mypath='C:/Users/w10007346/Dropbox/CNN/AugmentedImages/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in onlyfiles:
    fullpath=mypath+file
    img=mpimg.imread(fullpath)
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    mpimg.imsave(file,img_rescale)


