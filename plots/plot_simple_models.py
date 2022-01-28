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
time_grid = jnp.arange(0, 1+0.01, 0.01)
time_grid = time_grid*(2-time_grid)

#%% Loading and Plotting Bridge

step = 500
max_val = 6500

X = []
W = []
N = int(max_val/step)
for i in range(1,N):
    dat = jnp.load('../main/bridge_estimation/simple_models/'+model+'_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt']
    Wt = dat['Wt']
    X.append(Xt)
    W.append(Wt) 
    
    
plt.figure(figsize=(8,6))
a = jnp.arange(1/N, 1, 1/N)
plt.plot(time_grid, X[0][:,0], color='red', alpha=float(a[0]), label='$q_{1}(t)$')
plt.plot(time_grid, X[0][:,1], color='blue', alpha=float(a[0]), label='$q_{2}(t)$')
plt.plot(time_grid, X[0][:,2], color='black', alpha=float(a[0]), label='$q_{3}(t)$')
for i in range(1, N-1):
    plt.plot(time_grid, X[i][:,0], color='red', alpha=float(a[i]))
    plt.plot(time_grid, X[i][:,1], color='blue', alpha=float(a[i]))
    plt.plot(time_grid, X[i][:,2], color='black', alpha=float(a[i]))
    
plt.grid()
plt.legend()
plt.tight_layout()

max_val = 2500
X = []
W = []
N = int(max_val/step)
for i in range(1,N):
    dat = jnp.load('../main/bridge_estimation/simple_models/'+model+'_theta_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt']
    Wt = dat['Wt']
    theta = dat['theta']
    X.append(Xt)
    W.append(Wt) 
    
    
plt.figure(figsize=(8,6))
a = jnp.arange(1/N, 1, 1/N)
plt.plot(time_grid, X[0][:,0], color='red', alpha=float(a[0]), label='$q_{1}(t)$')
plt.plot(time_grid, X[0][:,1], color='blue', alpha=float(a[0]), label='$q_{2}(t)$')
plt.plot(time_grid, X[0][:,2], color='black', alpha=float(a[0]), label='$q_{3}(t)$')
for i in range(1, N-1):
    plt.plot(time_grid, X[i][:,0], color='red', alpha=float(a[i]))
    plt.plot(time_grid, X[i][:,1], color='blue', alpha=float(a[i]))
    plt.plot(time_grid, X[i][:,2], color='black', alpha=float(a[i]))
    
plt.grid()
plt.xlabel('$t$')
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

#%% Loading and Plotting Template

step = 50
max_val = 2500

X = []
W = []
N = int(2500/step)
for i in range(1,N):
    dat = jnp.load('../main/template_estimation/simple_models/'+model+'_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt']
    Wt = dat['Wt']
    X.append(Xt)
    W.append(Wt) 
    
I = X[0].shape[0]
obs = X[-1][:,-1,0:3]
plt.figure(figsize=(8,6))
plt.scatter(1*jnp.ones(I), obs[:,0], color='black', label='Observation')
plt.scatter(2*jnp.ones(I), obs[:,1], color='black')
plt.scatter(3*jnp.ones(I), obs[:,2], color='black')
for i in range(1, I):
    plt.scatter(1*jnp.ones(I), obs[:,0], color='black')
    plt.scatter(2*jnp.ones(I), obs[:,1], color='black')
    plt.scatter(3*jnp.ones(I), obs[:,2], color='black')
    
a = jnp.arange(1/N, 1, 1/N)
plt.scatter(1, X[i][0,0,0], color='red', alpha=float(a[0]), label='Template Estimation')
plt.scatter(2, X[i][0,0,1], color='red', alpha=float(a[0]))
plt.scatter(3, X[i][0,0,2], color='red', alpha=float(a[0]))
for i in range(1, N-1):
    plt.scatter(1, X[i][0,0,0], color='red', alpha=float(a[i]))
    plt.scatter(2, X[i][0,0,1], color='red', alpha=float(a[i]))
    plt.scatter(3, X[i][0,0,2], color='red', alpha=float(a[i]))
    
    
step = 50
max_val = 2500

X = []
W = []
N = int(2500/step)
for i in range(1,N):
    dat = jnp.load('../main/template_estimation/simple_models/'+model+'_theta_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt']
    Wt = dat['Wt']
    theta = dat['theta']
    X.append(Xt)
    W.append(Wt) 
    
I = X[0].shape[0]
obs = X[-1][:,-1,0:3]
plt.figure(figsize=(8,6))
plt.scatter(1*jnp.ones(I), obs[:,0], color='black', label='Observation')
plt.scatter(2*jnp.ones(I), obs[:,1], color='black')
plt.scatter(3*jnp.ones(I), obs[:,2], color='black')
for i in range(1, I):
    plt.scatter(1*jnp.ones(I), obs[:,0], color='black')
    plt.scatter(2*jnp.ones(I), obs[:,1], color='black')
    plt.scatter(3*jnp.ones(I), obs[:,2], color='black')
    
a = jnp.arange(1/N, 1, 1/N)
plt.scatter(1, X[i][0,0,0], color='red', alpha=float(a[i]), label='Template Estimation')
plt.scatter(2, X[i][0,0,1], color='red', alpha=float(a[i]))
plt.scatter(3, X[i][0,0,2], color='red', alpha=float(a[i]))
for i in range(1, N-1):
    plt.scatter(1, X[i][0,0,0], color='red', alpha=float(a[i]))
    plt.scatter(2, X[i][0,0,1], color='red', alpha=float(a[i]))
    plt.scatter(3, X[i][0,0,2], color='red', alpha=float(a[i]))
    
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

