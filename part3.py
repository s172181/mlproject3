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
from ex12_1_4 import mat2transactions, print_apriori_rules
from apyori import apriori

from basic import *


attributeNames = list(attributeNames)

# We will now transform the wine dataset into a binary format. Notice the changed attribute names:

Xbin, attributeNamesBin = binarize2(X, attributeNames)
print("X, i.e. the wine dataset, has now been transformed into:")
#print(Xbin)
#print(attributeNamesBin)



# Given the processed data in the previous exercise this becomes easy:
T = mat2transactions(Xbin,labels=attributeNamesBin)
rules = apriori(T, min_support=0.45, min_confidence=.6)
print_apriori_rules(rules)