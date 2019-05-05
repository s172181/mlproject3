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
from similarity import binarize2
from apyori import apriori

from basic import *

def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T

def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:        
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ) )
            y = ", ".join( list( o.items_add ) )
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
    return frules


attributeNames = list(attributeNames)

# We will now transform the wine dataset into a binary format. Notice the changed attribute names:

Xbin, attributeNamesBin = binarize2(X, attributeNames)
print("X, i.e. the dataset, has now been transformed into:")
#print(attributeNamesBin)

# Given the processed data in the previous exercise this becomes easy:
T = mat2transactions(Xbin,labels=attributeNamesBin)
rules = apriori(T, min_support=0.45, min_confidence=.6)
print_apriori_rules(rules)