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
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from sklearn.mixture import GaussianMixture

#from basic_classification_nsm import *


# Load the Energy csv data using the Pandas library
filename = 'energydata_complete_nsm_timeday_final_ordered.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
raw_data = df.get_values() 
cols = [1,21]

#Create the array y
y = raw_data[:, 21]
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
# Range of K's to try
KRange = range(1,10)
T = len(KRange)

covar_type = 'full'       # you can try out 'diag' as well
reps = 3                  # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans' # 'kmeans' or 'random'

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, 
                              n_init=reps, init_params=init_procedure,
                              tol=1e-3, reg_covar=1e-3).fit(X)
        
        # Get BIC and AIC
        BIC[t,] = gmm.bic(X)
        AIC[t,] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()
            

# Plot results

figure(1); 
plot(KRange, BIC,'-*b')
plot(KRange, AIC,'-xr')
plot(KRange, 2*CVE,'-ok')
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
show()


