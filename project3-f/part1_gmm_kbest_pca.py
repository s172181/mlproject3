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
cols = range(1, 28) 

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
Xmeans = X - np.ones((N,1))*X.mean(axis=0)

#normalize each attribute by further dividing each attribute by its standard deviation
for c in range(M):
   stdv = Xmeans[:,c].std()
   for r in range(N):
       Xmeans[r,c] = Xmeans[r,c]/stdv

#Calculate PCA
U,S,V = linalg.svd(Xmeans,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

#Projecting the data into the principal components
Z_pc = np.dot(Xmeans,V)
Z_pc_n = Z_pc[:,range(1,3)]

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
                              tol=1e-3, reg_covar=1e-3).fit(Z_pc_n)
        
        # Get BIC and AIC
        BIC[t,] = gmm.bic(Z_pc_n)
        AIC[t,] = gmm.aic(Z_pc_n)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = Z_pc_n[train_index]
            X_test = Z_pc_n[test_index]

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


