#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 00:46:03 2022

@author: root
"""

#%% Modules

import jax.numpy as jnp

import time

import integration_ode as inter

#%% Testing

def f_fun(t):
    
    return t**2

grid = jnp.linspace(5,6,100)
f_val = grid**2
t0 = time.time()
yT, yt = inter.integrator(f_fun, grid, method='euler')
t1 = time.time()
print(t1-t0)
t0 = time.time()
yT2, yt = inter.integrator(f_val, grid, method='euler')
t1 = time.time()
print(t1-t0)

t0 = time.time()
yT, yt = inter.integrator(f_fun, grid, method='trapez')
t1 = time.time()
print(t1-t0)
t0 = time.time()
yT2, yt = inter.integrator(f_val, grid, method='trapez')
t1 = time.time()
print(t1-t0)