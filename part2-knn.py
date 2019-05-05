#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 15:09:30 2019

@author: 
"""

import numpy as np
from matplotlib.pyplot import (figure, imshow, bar, title, xticks, yticks, cm,
                               subplot, show)
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

from basic import *

### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1./(D.sum(axis=1)/K)

# Sort the scores
i = density.argsort()
density = density[i]

# Plot k-neighbor estimate of outlier score (distances)
figure(3)
bar(range(20),density[:20])
title('KNN density: Outlier score')
# Plot possible outliers
figure(4)
for k in range(1,21):
    print('Index :{:2d}'.format(i[k]))
    print(X_or[i[k],:])
