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

n = 15
d = 2

a = 0.5
b = 2.0
phi = jnp.linspace(0, 2*jnp.pi, n, endpoint=False)
q0 = (jnp.vstack((a*jnp.cos(phi), b*jnp.sin(phi))).T)


a = 1.0
b = 2.0
phi = jnp.linspace(0, 2*jnp.pi, n, endpoint=False)
qT = (jnp.vstack((a*jnp.cos(phi), b*jnp.sin(phi))).T)

#%% Loading and Plotting Bridge

step = 500
max_val = 4000

X = []
W = []
N = int(max_val/step)
for i in range(1,N):
    dat = jnp.load('../main/bridge_estimation/ellipse_models/'+model+'_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt'].reshape(101,30,2)
    Wt = dat['Wt']
    X.append(Xt)
    W.append(Wt) 
    
    
plt.figure(figsize=(8,6))
a = jnp.arange(1/N, 1, 1/N)
plt.plot(X[0][-1,0:15,0], X[0][-1,0:15,1], color='red', alpha=float(a[0]), label='$q(t)$')
plt.plot(qT[:,0], qT[:,1], color='black', label='$q_{T}$')
for i in range(1, N-1):
    plt.plot(X[i][-1,0:15,0], X[i][-1,0:15,1], color='red', alpha=float(a[i]))
    
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
    dat = jnp.load('../main/bridge_estimation/ellipse_models/'+model+'_theta_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt'].reshape(101,30,2)
    Wt = dat['Wt']
    theta = dat['theta']
    X.append(Xt)
    W.append(Wt) 
    
    
plt.figure(figsize=(8,6))
a = jnp.arange(1/N, 1, 1/N)
plt.plot(X[0][-1,0:15,0], X[0][-1,0:15,1], color='red', alpha=float(a[0]), label='$q(t)$')
plt.plot(qT[:,0], qT[:,1], color='black', label='$q_{T}$')
for i in range(1, N-1):
    plt.plot(X[i][-1,0:15,0], X[i][-1,0:15,1], color='red', alpha=float(a[i]))
    
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
max_val = 2500

X = []
W = []
N = int(2500/step)
for i in range(1,N):
    dat = jnp.load('../main/template_estimation/ellipse_models/'+model+'_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt'].reshape(-1,101,30,2)
    Wt = dat['Wt']
    X.append(Xt)
    W.append(Wt) 
    
I = X[0].shape[0]
obs = X[-1][:,-1,0:15]
plt.figure(figsize=(8,6))
plt.plot(obs[0,:,0], obs[0,:,1], color='black', label='Observation')
for i in range(1, I):
    plt.plot(obs[i,:,0], obs[i,:,1], color='black', label='Observation')
    
a = jnp.arange(1/N, 1, 1/N)
plt.plot(X[0][0,0,0:15,0], X[0][0,0,0:15,1], color='red', alpha=float(a[0]), label='Template Estimation')
for i in range(1, N-1):
    plt.plot(X[i][0,0,0:15,0], X[i][0,0,0:15,1], color='red', alpha=float(a[0]), label='Template Estimation')

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
    dat = jnp.load('../main/template_estimation/ellipse_models/'+model+'_theta_iter_'+str(i*step)+'.npz')
    Xt = dat['Xt'].reshape(-1,101,30,2)
    Wt = dat['Wt']
    theta = dat['theta']
    X.append(Xt)
    W.append(Wt) 
    
I = X[0].shape[0]
obs = X[-1][:,-1,0:15]
plt.figure(figsize=(8,6))
plt.plot(obs[0,:,0], obs[0,:,1], color='black', label='Observation')
for i in range(1, I):
    plt.plot(obs[i,:,0], obs[i,:,1], color='black', label='Observation')
    
a = jnp.arange(1/N, 1, 1/N)
plt.plot(X[0][0,0,0:15,0], X[0][0,0,0:15,1], color='red', alpha=float(a[0]), label='Template Estimation')
for i in range(1, N-1):
    plt.plot(X[i][0,0,0:15,0], X[i][0,0,0:15,1], color='red', alpha=float(a[0]), label='Template Estimation')

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


