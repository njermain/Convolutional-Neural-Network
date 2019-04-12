# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:51:05 2019

@author: w10007346
"""

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = InceptionV3(weights='imagenet', include_top=False)
model.summary()

inception_feature_list = []
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

    inception_feature = model.predict(img_data)
    inception_feature_np = np.array(inception_feature)
    inception_feature_list.append(inception_feature_np.flatten())

len(inception_feature)

import pickle
with open('inception_feature', 'wb') as f:
      pickle.dump(inception_feature_list, f)

X=inception_feature_list[0:500]



pca=PCA(n_components=2)

principalComponents=pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
filenum=files[0:500]
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
DF['Fish ID Number']
DF['Reader 1 Age (KP)'].isnull().values.any()


scatter_x=np.array(DF['principal component 1'])
scatter_y=np.array(DF['principal component 2'])
age=pd.to_numeric(DF['Reader 1 Age (KP)'], downcast='integer')


fig, ax = plt.subplots()
for g in np.unique(age):
    i = np.where(age == g)
    ax.scatter(scatter_x[i], scatter_y[i], label=g)
ax.legend()
plt.show()



















