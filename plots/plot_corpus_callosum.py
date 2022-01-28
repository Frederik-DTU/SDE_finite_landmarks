#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 09:18:35 2022

@author: root
"""

#%% Modules used

import jax.numpy as jnp

import matplotlib.pyplot as plt

import scipy.io as io

#%% Hyper-parameters

model = 'tv'
time_grid = jnp.arange(0, 1+0.01, 0.01)
time_grid = time_grid*(2-time_grid)


data=io.loadmat('Data/dataM-corpora-callosa-ipmi-full.mat')
vi = data['vi'] # corpora callosa
Nobs = vi.shape[1]
N0 = int(vi.shape[0]/2-1)
qs = vi[0:-2,:].T.reshape((Nobs,N0,2))

    
n = N0
d = 2

q0 = qs[0]
qT = qs[1]
vT = qT

#%% Loading and Plotting Bridge

step = 500
max_val = 3000

X = []
W = []
N = int(max_val/step)
for i in range(1,N):
    dat = jnp.load('../main/bridge_estimation/corpus_callosum_models/'+model+'_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt'].reshape(101,2*n,d)
    Wt = dat['Wt']
    X.append(Xt)
    W.append(Wt) 
    
    
plt.figure(figsize=(8,6))
a = jnp.arange(1/N, 1, 1/N)
plt.plot(X[0][-1,0:n,0], X[0][-1,0:n,1], color='red', alpha=float(a[0]), label='$q(t)$')
plt.plot(qT[:,0], qT[:,1], color='black', label='$q_{T}$')
for i in range(1, N-1):
    plt.plot(X[i][-1,0:n,0], X[i][-1,0:n,1], color='red', alpha=float(a[i]))
    
plt.grid()
plt.legend()
plt.xlabel('$q_{1}(t)$')
plt.ylabel('$q_{2}(t)$')
plt.title(model+'-model')
plt.tight_layout()

max_val = 2500
X = []
W = []
N = int(max_val/step)
for i in range(1,N):
    dat = jnp.load('../main/bridge_estimation/corpus_callosum_models/'+model+'_theta_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt'].reshape(101,2*n,d)
    Wt = dat['Wt']
    theta = dat['theta']
    X.append(Xt)
    W.append(Wt) 
    
    
plt.figure(figsize=(8,6))
a = jnp.arange(1/N, 1, 1/N)
plt.plot(X[0][-1,0:n,0], X[0][-1,0:n,1], color='red', alpha=float(a[0]), label='$q(t)$')
plt.plot(qT[:,0], qT[:,1], color='black', label='$q_{T}$')
for i in range(1, N-1):
    plt.plot(X[i][-1,0:n,0], X[i][-1,0:n,1], color='red', alpha=float(a[i]))
    
plt.grid()
plt.legend()
plt.xlabel('$q_{1}(t)$')
plt.ylabel('$q_{2}(t)$')
plt.title(model+'-model')
plt.tight_layout()

plt.figure(figsize=(8,6))
plt.plot(jnp.arange(1,len(theta)+1), theta)
plt.ylim((0,2))
plt.grid()
plt.xlabel('$Iterations$')
plt.ylabel(r'$\theta$')
plt.tight_layout()

#%% Loading and Plotting Template

step = 50
max_val = 1950

n = 22
d = 2

X = []
W = []
N = int(max_val/step)
for i in range(1,N):
    dat = jnp.load('../main/template_estimation/corpus_callosum_models/'+model+'_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt'].reshape(-1,101,2*n,d)
    Wt = dat['Wt']
    X.append(Xt)
    W.append(Wt) 
    
I = X[0].shape[0]
obs = X[-1][:,-1,0:n]
plt.figure(figsize=(8,6))
plt.plot(obs[0,:,0], obs[0,:,1], color='black', label='Observation')
for i in range(1, I):
    plt.plot(obs[i,:,0], obs[i,:,1], color='black')
    
a = jnp.arange(1/N, 1, 1/N)
plt.plot(X[0][0,0,0:n,0], X[0][0,0,0:n,1], color='red', alpha=float(a[0]), label='Template Estimation')
for i in range(1, N-1):
    plt.plot(X[i][0,0,0:n,0], X[i][0,0,0:n,1], color='red', alpha=float(a[i]))

plt.grid()
plt.xlabel('$n$')
plt.ylabel('$q(t)$')
plt.title(str(model)+'-model')
plt.legend()
plt.tight_layout()
    
step = 50
max_val = 200

X = []
W = []
N = int(max_val/step)
for i in range(1,N):
    dat = jnp.load('../main/template_estimation/corpus_callosum_models/'+model+'_theta_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt'].reshape(-1,101,2*n,d)
    Wt = dat['Wt']
    theta = dat['theta']
    X.append(Xt)
    W.append(Wt) 
    
I = X[0].shape[0]
obs = X[-1][:,-1,0:15]
plt.figure(figsize=(8,6))
plt.plot(obs[0,:,0], obs[0,:,1], color='black', label='Observation')
for i in range(1, I):
    plt.plot(obs[i,:,0], obs[i,:,1], color='black')
    
a = jnp.arange(1/N, 1, 1/N)
plt.plot(X[0][0,0,0:n,0], X[0][0,0,0:n,1], color='red', alpha=float(a[0]), label='Template Estimation')
for i in range(1, N-1):
    plt.plot(X[i][0,0,0:n,0], X[i][0,0,0:n,1], color='red', alpha=float(a[i]))

plt.grid()
plt.xlabel('$n$')
plt.ylabel('$q(t)$')
plt.title(str(model)+'-model')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8,6))
plt.plot(jnp.arange(1,len(theta)+1), theta)
plt.ylim((0,2))
plt.grid()
plt.xlabel('$Iterations$')
plt.ylabel(r'$\theta$')
plt.tight_layout()


