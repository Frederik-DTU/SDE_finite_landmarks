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

def b_fun(t,x):
    
    return 0.5

def sigma_fun(t,x):
    
    return 2.0

def gbm_b(t,x):
    
    return 0.5*x

def gbm_sigma(t,x):
    
    return 0.5*x


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



    