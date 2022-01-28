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

#%% Loading and Plotting Bridge

X = []
W = []
N = int(10500/500)
for i in range(1,N):
    dat = jnp.load('../main/bridge_estimation/simple_models/'+model+'_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt']
    Wt = dat['Wt']
    X.append(Xt)
    W.append(Wt) 
    
    
plt.figure(figsize=(8,6))
a = jnp.arange(1/N, 1, 1/N)
plt.plot(time_grid, X[0][:,0], color='red', alpha=float(a[i]), label='$q_{1}(t)$')
plt.plot(time_grid, X[0][:,1], color='blue', alpha=float(a[i]), label='$q_{2}(t)$')
plt.plot(time_grid, X[0][:,2], color='black', alpha=float(a[i]), label='$q_{3}(t)$')
for i in range(1, N-1):
    plt.plot(time_grid, X[i][:,0], color='red', alpha=float(a[i]))
    plt.plot(time_grid, X[i][:,1], color='blue', alpha=float(a[i]))
    plt.plot(time_grid, X[i][:,2], color='black', alpha=float(a[i]))
    
plt.grid()
plt.legend()
plt.tight_layout()

#%% Loading and Plotting Template

X = []
W = []
N = int(10500/500)
for i in range(1,N):
    dat = jnp.load('../main/template_estimation/simple_models/'+model+'_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt']
    Wt = dat['Wt']
    X.append(Xt)
    W.append(Wt) 
    
I = X[0].shape[0]
plt.figure(figsize=(8,6))
a = jnp.arange(1/N, 1, 1/N)
for i in range(I):
    plt.scatter(X[-1][i,-1,0:3])

