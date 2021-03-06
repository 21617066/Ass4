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

# Define signals
signal1 = np.array([[ 1. ,  1.1,  0.9, 1.0, 0.0,  0.2,  0.1,  0.3,  3.4,  3.6,  3.5]])
signal2 = np.array([[0.8, 1.2, 0.4, 0.2, 0.15, 2.8, 3.6]])

# Collect training data together
toy_data = np.hstack([signal1, signal2])
toy_lengths = [11, 7]

# Create and fit HMM model to data
toy_hmm = HMM()
toy_hmm.fit(toy_data, toy_lengths, 3)

toy_means = [d.get_mean() for d in toy_hmm.dists]
toy_covs = [d.get_cov() for d in toy_hmm.dists]
print ('Transition probabilities: ')
print (toy_hmm.trans)
print ('Means: ')
print (toy_means)
print ('Covariances: ')
print (toy_covs)




# means = [1.0,0.19285714, 3.38]     
# covs = [0.01666667, 0.01459184, 0.0896]
# trans= np.array([[0.66666667, 0.33333333, 0.0,0.0],[0.0,0.71428571,0.28571429,0.0],[0.0,0.0,0.6,0.4],[1.0,0.0,0.0 , 0.0]])

# toy_hmm = HMM()
# dists = [Gaussian(mean=np.array([means[i]]), cov=np.array([[covs[i]]])) for i in range(len(covs))]
# toy_hmm.load(trans, dists)

samples, states = toy_hmm.sample()
print(samples)
print(states)

# signal3 = np.array([[ 0.9515792,   0.9832767,   1.04633007,  1.01464327,  0.98207072,  1.01116689,
#   0.31622856,  0.20819263,  3.57707616]])
# signal4 = np.array([[ 1,   0.6,   0.5 , 3.57707616]]) 

# print(toy_hmm.score(signal3, np.array([0, 0, 0, 0, 0, 1, 1, 1, 2])))
# print(toy_hmm.score(signal4, np.array([0,1,1,2])))

# scores = np.empty(0)
# ll = np.empty((3,1))
# seq = np.array([[0,0,1,2], [0,1,1,2], [0,1,2,2]])

# ll[0] = toy_hmm.score(signal4, seq[0])
# ll[1] = toy_hmm.score(signal4, seq[1])
# ll[2] = toy_hmm.score(signal4, seq[2])

# print("ll1: " + str(ll[0]) + " likelihood: " + str(np.exp(ll[0])))
# print("ll2: " + str(ll[1]) + " likelihood: " + str(np.exp(ll[1])))
# print("ll3: " + str(ll[2]) + " likelihood: " + str(np.exp(ll[2])))
# print("Max score(ll): " + str(np.max(ll)) + " with seq: " + str(seq[np.argmax(ll)]))

# ll_sum = np.log(np.exp(ll[0]) + np.exp(ll[1]) + np.exp(ll[2]))
# print("ll_total: " + str(ll_sum))

# print(toy_hmm.forward(signal4))






# signal5 = np.array([[ 0.9515792,   0.9832767,   1.04633007,  1.01464327,  1.98207072,  1.01116689,
#   0.31622856,  0.20819263,  2.57707616]])
# seq5, ll5 = toy_hmm.viterbi(signal5)
# print(seq5)
# print(ll5)

# signal6 = np.array([[ 1,   0.6,   1.9773 , 2.653997]]) 
# seq6, ll6 = toy_hmm.viterbi(signal6)
# print(seq6)
# print(ll6)

# toy_means = [d.get_mean() for d in toy_hmm.dists]
# toy_covs = [d.get_cov() for d in toy_hmm.dists]
# print ('Transition probabilities: ')
# print (toy_hmm.trans)
# print ('Means: ')
# print (toy_means)
# print ('Covariances: ')
# print (toy_covs)
