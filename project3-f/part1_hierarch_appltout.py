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
import matplotlib.pyplot as plt

#from basic_classification_nsm import *


# Load the Energy csv data using the Pandas library
filename = 'energydata_complete_nsm_timeday_final_ordered.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
raw_data = df.get_values() 
#cols = range(1, 28) 
cols = [1,21]

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

##############
#############

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'ward'
Metric = 'euclidean'

#hierarchical clustering,!!!
#Function linkage() forms
#a sample to sample distance matrix according to a given 
#distance metric, and creates
#the linkages between data points forming the hierarchical cluster tree
Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 6

#To compute a clustering, you can use the function fcluster
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=6
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()
