#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 00:43:30 2022

@author: root
"""

#%% Modules

import jax.numpy as jnp

import time

import backward_filtering as bf


#%% Testing

def grid_fun(t0,t1):
    
    return jnp.linspace(t0,t1,100000)

def beta_fun(t):
    
    return jnp.ones(2)*t

def B_fun(t):
    
    return jnp.eye(2)*t

def a_fun(t):
    
    return jnp.eye(2)

LT = jnp.array([1.0, 0.0])
SigmaT = jnp.eye(1)
vt = jnp.array([1.0])
muT = jnp.zeros(len(SigmaT))
time_grid = jnp.linspace(0,1,100000)



t0 = time.time()
Lt, Mt, mut, Ft, Ht = bf.lmmu_step(beta_fun, B_fun, a_fun, LT, SigmaT, muT, vt, 
                                time_grid, method='euler')
t1 = time.time()
print(t1-t0)
Lt2, Mt2, mut2, Ft2, Ht2 = bf.lmmu_step(beta_fun, B_fun, a_fun, LT, SigmaT, muT, vt, 
                                time_grid, method='odeint')

t = jnp.array([0.0, 0.5, 1.0])
Sigma_list = [jnp.eye(2), jnp.eye(1).reshape(1,1), jnp.eye(1).reshape(1,1)]
L_list = [jnp.eye(2), jnp.array([1.0, 0.0]).reshape(1,2), jnp.array([1.0, 0.0]).reshape(1,2)]
v_list = [jnp.array([1.0, 0.5]), jnp.array([2.0]), jnp.array([3.0])]
P_n = 100*jnp.eye(2)
t0 = time.time()
Lt_list, Mt_list, mut_list, Ft_list, Ht_list = bf.lmmu(beta_fun, B_fun, a_fun, P_n, 
                                                    L_list, Sigma_list, v_list, t, 
                                                    grid_fun, method='odeint')
t1 = time.time()
print(t1-t0)


t_points = jnp.array([0.0, 0.5, 1.0])
t0 = time.time()
Ht10, Ft10, ct10 = bf.hfc(beta_fun, B_fun, a_fun, P_n, L_list, Sigma_list, v_list, 
               t_points, grid_fun, method='odeint')
t1 = time.time()
print(t1-t0)


t0 = time.time()
Pt10, nut10, Ht10 = bf.pnu(beta_fun, B_fun, a_fun, P_n, L_list, Sigma_list, v_list, t_points, grid_fun, method='odeint')
t1 = time.time()
print(t1-t0)

HT_ = jnp.eye(2)
FT_ = jnp.ones(2)
cT_ = jnp.array([0.0])
t0 = time.time()
Ht, Ft, ct = bf.hfc_step(beta_fun, B_fun, a_fun, HT_, FT_, cT_, vt, 
                               time_grid, method='odeint')
t1 = time.time()
print(t1-t0)

Ht2, Ft2, ct2 = bf.hfc_step(beta_fun, B_fun, a_fun, HT_, FT_, cT_, vt, 
                               time_grid, method='euler')

PT_ = jnp.eye(2)
nuT_ = jnp.ones(2)
t0 = time.time()
Pt, nut, Ht = bf.pnu_step(beta_fun, B_fun, a_fun, PT_, nuT_, LT, SigmaT, vt, time_grid, method='odeint')
t1 = time.time()
print(t1-t0)

Pt2, nut2, Ht2 = bf.pnu_step(beta_fun, B_fun, a_fun, PT_, nuT_, LT, SigmaT, vt, time_grid, method='euler')
