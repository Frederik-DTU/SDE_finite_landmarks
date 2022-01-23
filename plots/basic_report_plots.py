#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 21:31:00 2022

@author: root
"""

#%% Modules

import jax.numpy as jnp

import matplotlib.pyplot as plt
import sim_sp as sp

#%% Plotting time interval change

grid = jnp.linspace(0,1,1000)
grid_transformed = grid*(2-grid)

fig, ax = plt.subplots(1,2,figsize=(8,6))
ax[0].plot(grid, grid_transformed, color='blue')
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$t(2-t)$')
ax[0].set_title('Transformation of Time Grid')
ax[0].grid()
plt.tight_layout()

ax[1].hist(grid_transformed, density=False, color='blue')
ax[1].grid()
ax[1].set_xlabel('$t$')
ax[1].set_title('Histogram for Transformation')
plt.tight_layout()

#%% Plotting theta kernel and probability distribution

def q_sample(theta):
    
    sigma_theta = 1
    theta_circ = jnp.exp(sp.sim_normal(mu=theta, sigma=sigma_theta))
    
    return theta_circ

def q_prob(theta):
    
    if theta >= 0.1:
        return 0.1*(theta**(-2))
    else:
        return 0
    
    
fig, ax = plt.subplots(1,2,figsize=(8,6))
theta = 1.0
theta_circ = jnp.zeros(1000)
for i in range(1000):
    theta_circ = theta_circ.at[i].set(q_sample(theta)[0])

ax[1].hist(theta_circ, density=False, color='blue', bins=20)
ax[1].grid()
ax[1].set_xlabel(r'$\theta^{\circ}$')
ax[1].set_title(r'Histogram for Markov kernel with $\theta=1$')
plt.tight_layout()    
    
    
zero = jnp.zeros(10)
zero_theta = jnp.linspace(0,0.1,10)
theta = jnp.linspace(0.1, 2, 1000)
p = 0.1*(theta**(-2))

ax[0].plot(theta, p, color='red')
ax[0].plot(zero_theta, zero, color='red')
ax[0].scatter(0.1, 0, s=20, facecolors='none', edgecolors='r')
ax[0].set_xlabel(r'$\theta$')
ax[0].set_ylabel(r'$p(\theta)$')
ax[0].set_title(r"Pareto Distribution with $\alpha=1$ and $x_m=0.1$")
ax[0].grid()
plt.tight_layout()

    
    