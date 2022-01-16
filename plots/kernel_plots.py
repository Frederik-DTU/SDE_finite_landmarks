#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 22:38:30 2022

@author: root
"""

#%% Modules

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


#%% Kernels

#%% Data

# number of points
N = 100

# generate points for class 1
r = np.random.rand(N, 1)
T = 2*np.pi*np.random.rand(N, 1)
points = r*np.exp(1j*T)
class1 = np.hstack((np.real(points), np.imag(points)))

r = np.random.rand(N, 1) + 1.5
T = 2*np.pi*np.random.rand(N, 1)
points = r*np.exp(1j*T)
class2 = np.hstack((np.real(points), np.imag(points)))

# stack data matrix
X = np.vstack((class1, class2))
N *= 2

fig, ax = plt.subplots(figsize=(8,6))
ax.grid()
ax.scatter(class1[:, 0], class1[:, 1], color='b', label='Group 1')
ax.scatter(class2[:, 0], class2[:, 1], color='r', label='Group 2')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Data')
ax.legend()

#%% Polynomial kernel

# calculate the homogenous polynomial kernel and plot
d = [1,5,10]
K = np.empty((len(d), N, N))
for r in range(len(d)):
    for i in range(N):
        for j in range(N):
            K[r, i, j] = np.dot(X[i, :], X[j, :])**r

# plot the K matrix on a color scale
fig, ax = plt.subplots(3, 3, figsize=(8,6))

im = ax[0,0].imshow(K[0], vmin=np.min(K[0]), vmax=np.max(K[0]))
divider = make_axes_locatable(ax[0,0])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[0,0].set_xlabel('d='+str(d[0]))
ax[0,0].get_xaxis().set_ticks([])
ax[0,0].get_yaxis().set_ticks([])
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([])

im = ax[0,1].imshow(K[1], vmin=np.min(K[1]), vmax=np.max(K[1]))
divider = make_axes_locatable(ax[0,1])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[0,1].set_xlabel('d='+str(d[1]))
ax[0,1].get_xaxis().set_ticks([])
ax[0,1].get_yaxis().set_ticks([])
ax[0,1].set_title('Polynomial Kernel')
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([])

im = ax[0,2].imshow(K[2], vmin=np.min(K[2]), vmax=np.max(K[2]))
divider = make_axes_locatable(ax[0,2])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[0,2].set_xlabel('d='+str(d[2]))
ax[0,2].get_xaxis().set_ticks([])
ax[0,2].get_yaxis().set_ticks([])
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([])

plt.tight_layout()

#%% Gaussian kernel

# calculate the homogenous polynomial kernel and plot
sigma = [0.01,1,100]
K = np.empty((len(d), N, N))
for r in range(len(sigma)):
    for i in range(N):
        for j in range(N):
            K[r, i, j] = np.exp(- np.linalg.norm(X[i, :]-X[j, :])**2 / (2*sigma[r]))

im = ax[1,0].imshow(K[0], vmin=np.min(K[0]), vmax=np.max(K[0]))
divider = make_axes_locatable(ax[1,0])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[1,0].set_xlabel('τ='+str(sigma[0]))
ax[1,0].get_xaxis().set_ticks([])
ax[1,0].get_yaxis().set_ticks([])
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([])

im = ax[1,1].imshow(K[1], vmin=np.min(K[1]), vmax=np.max(K[1]))
divider = make_axes_locatable(ax[1,1])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[1,1].set_xlabel('τ='+str(sigma[1]))
ax[1,1].get_xaxis().set_ticks([])
ax[1,1].get_yaxis().set_ticks([])
ax[1,1].set_title('Gaussian Kernel')
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([])

im = ax[1,2].imshow(K[2], vmin=np.min(K[2]), vmax=np.max(K[2]))
divider = make_axes_locatable(ax[1,2])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[1,2].set_xlabel('τ='+str(sigma[2]))
ax[1,2].get_xaxis().set_ticks([])
ax[1,2].get_yaxis().set_ticks([])
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([])

plt.tight_layout()

#%% Exponential kernel

# calculate the homogenous polynomial kernel and plot
sigma = [0.01,1,100]
K = np.empty((len(d), N, N))
for r in range(len(sigma)):
    for i in range(N):
        for j in range(N):
            K[r, i, j] = np.exp(- np.linalg.norm(X[i, :]-X[j, :]) / (sigma[r]))

im = ax[2,0].imshow(K[0], vmin=np.min(K[0]), vmax=np.max(K[0]))
divider = make_axes_locatable(ax[2,0])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[2,0].set_xlabel('$\sigma$='+str(sigma[0]))
ax[2,0].get_xaxis().set_ticks([])
ax[2,0].get_yaxis().set_ticks([])
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([])

im = ax[2,1].imshow(K[1], vmin=np.min(K[1]), vmax=np.max(K[1]))
divider = make_axes_locatable(ax[2,1])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[2,1].set_xlabel('$\sigma$='+str(sigma[1]))
ax[2,1].get_xaxis().set_ticks([])
ax[2,1].get_yaxis().set_ticks([])
ax[2,1].set_title('Exponential Kernel')
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([])

im = ax[2,2].imshow(K[2], vmin=np.min(K[2]), vmax=np.max(K[2]))
divider = make_axes_locatable(ax[2,2])
cax = divider.append_axes('right', size='5%', pad=0.05)
ax[2,2].set_xlabel('$\sigma$='+str(sigma[2]))
ax[2,2].get_xaxis().set_ticks([])
ax[2,2].get_yaxis().set_ticks([])
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([])

plt.tight_layout()
