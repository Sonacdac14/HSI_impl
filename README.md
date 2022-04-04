# HSI_Classification
In this Hyperspectral image classification implementation done by Indian_ pines datasets.
# libraries import
import numpy as np
import sklearn
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
# import the datasets
indian_pines = sio.loadmat(r"C:\Users\DELL\Downloads\archive (6)\Indian_pines_corrected.mat")
print(indian_pines)
# list vise classify the data
indian_pines_key=list(indian_pines.keys())
indian_pines_IN = (indian_pines[indian_pines_key[3]])
print(indian_pines_IN)
# print the data value
print(indian_pines_IN.shape)
# plotting the data sets
fig = plt.figure(figsize = (10,10))
plt.imshow(indian_pines_IN[:,:,2], interpolation='nearest')
plt.show()
indian_pine_data = indian_pines_IN.reshape(np.prod(indian_pines_IN.shape[:2]),np.prod(indian_pines_IN.shape[2:]))
# New shape of the data is
print(indian_pine_data.shape)
print(indian_pine_data)
from sklearn.preprocessing import StandardScaler
indian_pine_data = StandardScaler().fit_transform(indian_pine_data)
print(indian_pine_data.shape)
print(indian_pine_data)
# used PCA as a dimensionality reduction-
from sklearn.decomposition import PCA
pca_decompostn = PCA(n_components=40)
indian_pine_data_pca = pca_decompostn.fit_transform(indian_pine_data)
print(pca_decompostn.explained_variance_ratio_)
print(indian_pine_data_pca.shape)
indian_pine_data_pca_new = indian_pine_data_pca.reshape(145,145,40)
print(indian_pine_data_pca_new.shape)
fig = plt.figure(figsize = (10,10))
plt.imshow(indian_pine_data_pca_new[:,:,1], interpolation='nearest')
plt.show()



