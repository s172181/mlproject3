#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:11:55 2019

@author: alejuliet
"""
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import numpy as np
import pandas as pd
import xlrd
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm)
import scipy.linalg as linalg
from sklearn import model_selection
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

#from basic_classification_nsm import *


# Load the Energy csv data using the Pandas library
filename = 'energydata_complete_nsm_timeday_final_ordered.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
raw_data = df.get_values() 
cols = [1,2]
#cols = range(1,27)

#Create the array y
y = raw_data[:, 28]
y = np.array(y, dtype=np.float)
#Create the class array Y
classNames = np.unique(y)

#Get the attributes names
attributeNames = np.asarray(df.columns[cols])

#columns: from Appliances until Tdewpoint
X = raw_data[:, cols]
N, M = X.shape
classnames = ["22:30-7","7-10","10-17","17-22:30"]


#Calculate vector of means
X = np.array(X, dtype=np.float)
#Standardize
X = X - np.ones((N,1))*X.mean(axis=0)
for c in range(M):
   stdv = X[:,c].std()
   for r in range(N):
       X[r,c] = X[r,c]/stdv

#####################
# exercise 11.1.1

# Number of clusters
K = 5
cov_type = 'full' # e.g. 'full' or 'diag'

# define the initialization procedure (initial value of means)
initialization_method = 'kmeans'#  'random' or 'kmeans'
# random signifies random initiation, kmeans means we run a K-means and use the
# result as the starting point. K-means might converge faster/better than  
# random, but might also cause the algorithm to be stuck in a poor local minimum 

# type of covariance, you can try out 'diag' as well
reps = 3
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps, 
                      tol=1e-9, reg_covar=1e-9, init_params=initialization_method).fit(X)
cls = gmm.predict(X)    
# extract cluster labels
cds = gmm.means_        
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type.lower() == 'diag':
    new_covs = np.zeros([K,M,M])    
    
    count = 0    
    for elem in covs:
        temp_m = np.zeros([M,M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results:
figure(figsize=(14,9))
clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
show()

## In case the number of features != 2, then a subset of features most be plotted instead.
#figure(figsize=(14,9))
#idx = [0,1] # feature index, choose two features to use as x and y axis in the plot
#clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
#show()
