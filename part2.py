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

### K-nearest neigbor average relative density
# Compute the average relative density
# Neighbor to use:
K = 5

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1./(D.sum(axis=1)/K)
avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)


# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

# Plot k-neighbor estimate of outlier score (distances)
figure(5)
bar(range(20),avg_rel_density[:20])
title('KNN average relative density: Outlier score')
# Plot possible outliers
figure(6)
for k in range(1,21):
    print('Index :{:2d}'.format(i_avg_rel[k]))
    print(X_or[i_avg_rel[k],:])