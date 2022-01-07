#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 00:05:40 2022

@author: root
"""

#%% Modules used

import jax.numpy as jnp

import matplotlib.pyplot as plt

from plot_landmarks import plot_landmarks

#%% Plots

plt_lm = plot_landmarks()
time_grid = jnp.arange(0, 1+0.001, 0.001)
t = time_grid*(2-time_grid)


val = jnp.load('1d_saved/Wt_Xt_500.npz')
Xt = val['Xt']
qt = Xt[:,0:3].reshape(1,-1,3)
pt = Xt[:,3:].reshape(1,-1,3)
for i in range(1, 40):
    
    idx = (i+1)*500
    val = jnp.load('1d_saved/Wt_Xt_'+str(idx)+'.npz')
    Xt = val['Xt']
    qt = jnp.vstack((qt, Xt[:,0:3].reshape(1,-1,3)))
    pt = jnp.vstack((pt, Xt[:,3:].reshape(1,-1,3)))

fig, ax = plt.subplots(1, 2, figsize = (8,6), sharex = False)

t0 = jnp.ones_like(qt[0,0])*t[0]
T = jnp.ones_like(qt[0,0])*t[-1]

alphas = jnp.linspace(1/40,1,40)
for i in range(len(alphas)):
    ax[0].plot(t, qt[i,:,0].T, color='black', alpha=float(alphas[i]))
for i in range(len(alphas)):
    ax[0].plot(t, qt[i,:,1].T, color='orange', alpha=float(alphas[i]))
for i in range(len(alphas)):
    ax[0].plot(t, qt[i,:,2].T, color='blue', alpha=float(alphas[i]))
ax[0].scatter(x=T, y=qt[0,-1], marker='o', s=40, color='navy', label='qT')
ax[0].scatter(x=t0, y=qt[0,0], marker='o', s=40, color='cyan', label='q0')
ax[0].set(title='Landmarks', ylabel = '', xlabel='time')
ax[0].grid()
ax[0].legend()

alphas = jnp.linspace(1/40,1,40)
for i in range(len(alphas)):
    ax[1].plot(t, pt[i,:,0].T, color='black', alpha=float(alphas[i]))
for i in range(len(alphas)):
    ax[1].plot(t, pt[i,:,1].T, color='orange', alpha=float(alphas[i]))
for i in range(len(alphas)):
    ax[1].plot(t, pt[i,:,2].T, color='blue', alpha=float(alphas[i]))
ax[1].yaxis.tick_right()
ax[1].set(title='Momentum', ylabel='', xlabel='time')
ax[1].grid()
ax[1].legend()
    
plt.tight_layout()
plt.suptitle('Hallo')