# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:51:05 2019

@author: w10007346
"""

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet', include_top=False)
model.summary()

vgg16_feature_list = []
main_dir='C:/Users/w10007346/Dropbox/CNN/Unsupervised_Set/'
from os import listdir
from os.path import isfile, join

files=[f for f in listdir(main_dir) if isfile(join(main_dir, f))]

for file in files:
    fullpath=main_dir+file
    img = image.load_img(fullpath, target_size=(800, 800))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)
    vgg16_feature_list.append(vgg16_feature_np.flatten())

len(vgg16_feature_list)

import pickle
with open('three_age_list', 'wb') as f:
      pickle.dump(vgg16_feature_list, f)

X=vgg16_feature_list


pca=PCA(n_components=2)

principalComponents=pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
filenum=files
filenums=[w.replace('.JPG', '') for w in filenum]

principalDf['Fish ID Number']=filenums
principalDf.head()

# get ages from csv
al=pd.read_csv('C:/Users/w10007346/Dropbox/CNN/al_scales.csv')
ms=pd.read_csv('C:/Users/w10007346/Dropbox/CNN/ms_scales.csv')
tx=pd.read_csv('C:/Users/w10007346/Dropbox/CNN/tx_scales.csv')
la=pd.read_csv('C:/Users/w10007346/Dropbox/CNN/la_scales.csv')
agedf=al.append([ms,tx,la], ignore_index=True)
agedf.head()
agedf['Fish ID Number'].isnull().values.any()
agedf['Fish ID Number'].describe()
agedf['Age'].describe()
agedf['Age'].isnull().values.any()
agedf['Reader 1 Age (KP)'].describe()

# join PCA dataframe with ages
DF=pd.merge(principalDf,agedf, how='left', on='Fish ID Number')
DF.columns.values
DF['Reader 1 Age (KP)'].head()
DF['Fish ID Number'].head()
DF['Reader 1 Age (KP)'].isna().sum() 
DF=DF[np.isfinite(DF['Reader 1 Age (KP)'])]
DF['Reader 1 Age (KP)'].isna().sum() 

scatter_x=np.array(DF['principal component 1'])
scatter_y=np.array(DF['principal component 2'])
age=pd.to_numeric(DF['Reader 1 Age (KP)'], downcast='integer')


fig, ax = plt.subplots()
for g in np.unique(age):
    i = np.where(age == g)
    ax.scatter(scatter_x[i], scatter_y[i], label=g)
ax.legend()
plt.show()
fig.savefig('VGG16_PCA.png', dpi=800, bbox_inches="tight")


len(labels)
len(scatter_x)

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3)
kmeans.fit(X)
labels=kmeans.labels_

# use x-y points from before I got rid of samples without ages
len(scatter_x)
scatter_x=np.array(principalDf['principal component 1'])
scatter_y=np.array(principalDf['principal component 2'])
fig, ax = plt.subplots()
for g in np.unique(labels):
    i = np.where(labels == g)
    ax.scatter(scatter_x[i], scatter_y[i], label=g)
ax.legend()
plt.show()
fig.savefig('VGG16_PCA_Kmeans.png', dpi=800, bbox_inches="tight")

















