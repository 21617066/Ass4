#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 11:39:13 2022

@author: epep
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
from sklearn.metrics import confusion_matrix
from hmm_class import HMM
from gaussian import Gaussian

means = [1.0,0.19285714, 3.38]     
covs = [0.01666667, 0.01459184, 0.0896]
trans= np.array([[0.66666667, 0.33333333, 0.0,0.0],[0.0,0.71428571,0.28571429,0.0],[0.0,0.0,0.6,0.4],[1.0,0.0,0.0 , 0.0]])

toy_hmm = HMM()
dists = [Gaussian(mean=np.array([means[i]]), cov=np.array([[covs[i]]])) for i in range(len(covs))]
toy_hmm.load(trans, dists)

samples, states = toy_hmm.sample()
print(samples)
print(states)

signal3 = np.array([[ 0.9515792,   0.9832767,   1.04633007,  1.01464327,  0.98207072,  1.01116689,
  0.31622856,  0.20819263,  3.57707616]])
signal4 = np.array([[ 1,   0.6,   0.5 , 3.57707616]]) 

print(toy_hmm.score(signal3, np.array([0, 0, 0, 0, 0, 1, 1, 1, 2])))
print(toy_hmm.score(signal4, np.array([0,1,1,2])))

# toy_means = [d.get_mean() for d in toy_hmm.dists]
# toy_covs = [d.get_cov() for d in toy_hmm.dists]
# print ('Transition probabilities: ')
# print (toy_hmm.trans)
# print ('Means: ')
# print (toy_means)
# print ('Covariances: ')
# print (toy_covs)
