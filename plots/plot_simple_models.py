#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 09:18:35 2022

@author: root
"""

#%% Modules used

import jax.numpy as jnp

import matplotlib.pyplot as plt

#%% Hyper-parameters

model = 'tv'
step = 500
time_grid = jnp.arange(0, 1+0.01, 0.01)
time_grid = time_grid*(2-time_grid)

#%% Loading

X = []
W = []
N = int(20000/500)
for i in range(1,N):
    dat = jnp.load('../main/bridge_estimation/simple_models/'+model+'_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt']
    Wt = dat['Wt']
    X.append(Xt)
    W.append(Wt) 
    
plt.figure(figsize=(8,6))
for i in range(1,N):
    plt.plot(time_grid, X[0][:,0:3])




dat2 = jnp.load('../main/template_estimation/simple_models/tv_iter_5000.npz')