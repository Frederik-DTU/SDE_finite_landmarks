#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 00:36:26 2022

@author: root
"""

#%% Modules

import jax.numpy as jnp

import time

import diffusion_bridges as db

#%% Testing

q0 = jnp.array([1.0])
p0_init = jnp.array([2.0])
vT = jnp.array([4.0])
SigmaT = jnp.eye(1)*(10**(-6))
LT = jnp.array([1.0, 0.0]).reshape(1,2)
time_grid = jnp.linspace(0,1,10000)
time_grid = time_grid*(2-time_grid)
def b_fun(t,x, theta=None):
    
    return 0.5*x

def sigma_fun(t,x, theta=None):
    
    return 0.5*jnp.eye(2)*x

def sigmatilde_fun(t, theta=None):
    
    return 0.5*jnp.eye(2)*vT

def Btilde_fun(t, theta=None):
    
    return 0.5*jnp.eye(2)

def betatilde_fun(t, theta=None):
    
    return jnp.zeros(2)

def pi_prob(x,y):
    
    return 1.0

def q_sample(x):
    
    return 1.0

def q_prob(x,y):
    
    return 1.0

t0 = time.time()
Xt = db.landmark_segment(q0,
                  p0_init,
                  vT,
                  SigmaT,
                  LT,
                  b_fun,
                  sigma_fun,
                  betatilde_fun,
                  Btilde_fun,
                  sigmatilde_fun,
                  pi_prob,
                  time_grid,
                  max_iter= 10,
                  eta=0.98,#0.98
                  delta=0.01, #0.01
                  theta=1.0,
                  q_sample = q_sample,
                  q_prob = q_prob
                  )
t1 = time.time()
print(t1-t0)
