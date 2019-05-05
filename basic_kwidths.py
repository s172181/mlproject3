#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:20:15 2019

@author: 
    Alechandrina Pereira s172181
    Maciej Maj, s171706
    Sushruth Bangre, s190021
"""

import numpy as np
import pandas as pd
import xlrd
from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show
from sklearn import preprocessing
import numpy as np
from matplotlib.pyplot import (figure, imshow, bar, title, xticks, yticks, cm,
                               subplot, show)
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

#Normalization function
def normalize(column):
    upper = column.max()
    lower = column.min()
    y = (column - lower)/(upper-lower)
    return y

# Load the Energy csv data using the Pandas library
filename = 'energydata_complete_nsm.csv'
df = pd.read_csv(filename)
# Pandas returns a dataframe, (df) which could be used for handling the data.
raw_data = df.get_values() 

##
#Creating X and y
##
cols = range(2, 28) 
X = raw_data[:, cols]
X = np.array(X, dtype=np.float)
y = raw_data[:, 1]
y = np.array(y, dtype=np.float)
N, M = X.shape

#Get the attributes names
attributeNames = np.asarray(df.columns[cols])

####
#Normalization
####
#Standardize
X = X - np.ones((N,1))*X.mean(axis=0)
for c in range(M):
   stdv = X[:,c].std()
   for r in range(N):
       X[r,c] = X[r,c]/stdv
       
#normalize each attribute 
for c in range(M):
   X[:,c] = normalize(X[:,c])
   


# Estimate the optimal kernel density width, by leave-one-out cross-validation
### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
   print('Fold {:2d}, w={:f}'.format(i,w))
   density, log_density = gausKernelDensity(X,w)
   logP[i] = log_density.sum()
   
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))
