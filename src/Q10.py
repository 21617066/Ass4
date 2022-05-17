#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:09:57 2022

@author: epep
"""
import numpy as np

signal = np.array([[0,1,0]])
trans = np.array([[0.62, 0.38, 0],
                  [0.14, 0.86, 0],
                  [0.9, 0.1, 0]])

dists = np.array([[0.29, 0.71],
                  [0.78, 0.22]])

T = signal.shape[1]
N = 2

# Initialize log alpha
l_alpha = np.full((N+1,T+1), -np.inf)
l_alpha[-1,-1] = 0
for t in range(T):
    for j in range(N):
        ll_xs = np.log(dists[j, signal[:,t]])

        ll_sum = -np.inf # Initialize log sum
        for i in range(-1,N):
            l_a = np.log(trans[i,j])
            ll_sum = np.logaddexp(ll_sum, l_a + l_alpha[i, t-1])
        l_alpha[j,t] = ll_xs + ll_sum

ll = -np.inf # Initialize log sum
for j in range(N):
    # l_a = np.log(trans[j,N])
    ll = np.logaddexp(ll, l_alpha[j,T-1])