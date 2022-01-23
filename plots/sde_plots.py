#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 23:22:34 2022

@author: root
"""

#%% Sources


#%% Modules

import jax.numpy as jnp

import matplotlib.pyplot as plt

import sim_sp as sp

#%% Functions

vT = 1.0
sigma_v = 100
T = 1.0
sigma = 2.0
mu = 0.5

mu_gbm = 0.5
sigma_gbm = 0.5
def b_fun(t,x):
    
    return mu

def sigma_fun(t,x):
    
    return sigma

def gbm_b(t,x):
    
    return mu_gbm*x

def gbm_sigma(t,x):
    
    return sigma_gbm*x

def b_fun_guided(t,x):
    
    term1 = sigma_v**2+sigma**2*(T-t)
    term2 = vT-x-mu*(T-t)
    
    return b_fun(t,x)+term2/term1*sigma_fun(t,x)**2

def gbm_b_gp(t,x):
    
    term1 = sigma_v**2+sigma_gbm**2*(T-t)
    term2 = jnp.log(vT)-jnp.log(x)-(mu_gbm-sigma_gbm/2)*(T-t)
    
    return mu_gbm-sigma_gbm/2+term2/term1*sigma_gbm**2*x+sigma_gbm**2/2*x


#%% dXt=mu*dt+sigma*dWt

grid = jnp.linspace(0,1,1000)
Wt = sp.sim_Wt(grid, dim=1, simulations=10)
x0 = jnp.array(0.5).reshape(1)

Xt_const = jnp.zeros((10, 1000))
for i in range(10):
    Xt_const = Xt_const.at[i].set(sp.sim_sde_euler(x0, b_fun, sigma_fun, Wt[i], grid))
    
plt.figure(figsize=(8,6))
plt.plot(grid, Xt_const.T, color='black')
plt.xlabel('t')
plt.ylabel('$X_t$')
plt.title('$dX_t=0.5dt+2.0dW_t$')
plt.grid()
plt.tight_layout()

Xt_const_gp = jnp.zeros((10, 1000))
for i in range(10):
    Xt_const_gp = Xt_const_gp.at[i].set(sp.sim_sde_euler(x0, b_fun_guided, sigma_fun, Wt[i], grid))

plt.figure(figsize=(8,6))
plt.plot(grid, Xt_const_gp.T, color='black')
plt.scatter(grid[0], x0, marker='o', s=40, color='cyan', label='$x_0$')
plt.scatter(grid[-1], vT, marker='o', s=40, color='red', label='$v_T$')
plt.xlabel('t')
plt.ylabel('$X_t$')
plt.title('Diffusion Bridge for $dX_t=0.5dt+2.0dW_t$ with $\sigma_v=100$')
plt.grid()
plt.legend()
plt.tight_layout()

#%% dXt=mu*Xt*dt+sigma*Xt*dWt

grid = jnp.linspace(0,1,1000)
Wt = sp.sim_Wt(grid, dim=1, simulations=10)
x0 = jnp.array(0.5).reshape(1)

Xt_gbm = jnp.zeros((10, 1000))
for i in range(10):
    Xt_gbm = Xt_gbm.at[i].set(sp.sim_sde_euler(x0, gbm_b, gbm_sigma, Wt[i], grid))
    
plt.figure(figsize=(8,6))
plt.plot(grid, Xt_gbm.T, color='black')
plt.xlabel('t')
plt.ylabel('$X_t$')
plt.title('$dX_t=0.5X_tdt+0.5X_tdW_t$')
plt.grid()
plt.tight_layout()

Xt_gbm_gp = jnp.zeros((10, 1000))
for i in range(10):
    Xt_gbm_gp = Xt_gbm_gp.at[i].set(sp.sim_sde_euler(x0, gbm_b_gp, gbm_sigma, Wt[i], grid))

plt.figure(figsize=(8,6))
plt.plot(grid, Xt_gbm_gp.T, color='black')
plt.scatter(grid[0], x0, marker='o', s=40, color='cyan', label='$x_0$')
plt.scatter(grid[-1], vT, marker='o', s=40, color='red', label='$v_T$')
plt.xlabel('t')
plt.ylabel('$X_t$')
plt.title('Diffusion Bridge for $dX_t=0.5X_tdt+0.5X_tdW_t$ with $\sigma_v=100$')
plt.grid()
plt.legend()
plt.tight_layout()
