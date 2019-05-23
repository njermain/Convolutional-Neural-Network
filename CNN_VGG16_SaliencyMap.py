# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:06:28 2019

@author: w10007346
"""

# load a saved model
from keras.models import load_model
import os
os.chdir('C:/Users/w10007346/Dropbox/CNN')
saved_model = load_model('VGG16_FT.h5')
saved_model.summary()

#### Saliency Map #######


import matplotlib.pyplot as plt
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
import matplotlib.image as mpimg
import scipy.ndimage as ndimage

# read image to apply saliency map to
img=mpimg.imread('Yourimage.jpg')
plt.imshow(img)
plt.savefig('OrigScale.png', dpi=300)

# identify layer to calculate gradient 
layer_idx = utils.find_layer_idx(saved_model, 'dense_12')
# switch activation to linear
saved_model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(saved_model)

# compute gradient
grads = visualize_saliency(model, layer_idx, filter_indices=None, 
                           seed_input= img, backprop_modifier=None, \
                           grad_modifier="absolute")
plt.imshow(grads)
plt.savefig('SalMapRaw.png', dpi=300)


# Overlay on original image and smooth using a gaussian filter
gaus = ndimage.gaussian_filter(grads[:,:,2], sigma=5) 
plt.imshow(img)
plt.imshow(gaus, alpha=.7)
plt.axis('off')

plt.savefig('SalMapScale.png', dpi=300) 