#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 00:47:48 2022

@author: root
"""

#%% Modules

import jax.numpy as jnp

import time

import landmark_models as lm

#%% Testing

#Kernel function
def k(x:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
    
    theta = 1.0
    
    return jnp.exp(-jnp.dot(x,x)/((2*theta)**2))

#Kernel gradient
def grad_k(x:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
    
    theta = 1.0
    
    return (theta**(-2))*k(x,theta)*x

q0 = jnp.array([-0.5, 0.0, 0.1]).reshape(3,1)
p0 = jnp.array([-0.1, 0.0, 0.1]).reshape(3,1)
qT = jnp.array([-0.5, 0.2, 1.0]).reshape(3,1)
vT = qT

t0 = time.time()
dH = lm.dH_dq(q0,p0, grad_k, None)
t1 = time.time()
print(t1-t0)

t0 = time.time()
dH = lm.dH_dp(q0,p0, grad_k, None)
t1 = time.time()
print(t1-t0)

lmbda = 0.5
gamma = jnp.array([1.,2,4.])
ms_drift, ms_diffusion = lm.ms_model(3, 1, k, grad_k, lmbda, gamma)
x = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))

t0 = time.time()
drift = ms_drift(0,x,None)
t1 = time.time()
print(t1-t0)

t0 = time.time()
diffusion = ms_diffusion(0,x,None)
t1 = time.time()
print(t1-t0)

ms_betatilde, ms_Btilde, ms_diffusion_tilde = lm.ms_auxillary_model(3, 1, k, grad_k, lmbda, gamma, qT)
x = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))

t0 = time.time()
drift = ms_betatilde(0,None)
t1 = time.time()
print(t1-t0)

t0 = time.time()
diffusion = ms_Btilde(0,None)
t1 = time.time()
print(t1-t0)

t0 = time.time()
ms_diffusion_tilde(0,None)
t1 = time.time()
print(t1-t0)

gamma = jnp.array([1.,2,4.])
tv_drift, tv_diffusion = lm.tv_model(3, 1, k, grad_k, gamma)
x = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))

t0 = time.time()
drift = tv_drift(0,x,None)
t1 = time.time()
print(t1-t0)

t0 = time.time()
diffusion = tv_diffusion(0,x,None)
t1 = time.time()
print(t1-t0)

tv_betatilde, tv_Btilde, tv_diffusion_tilde = lm.tv_auxillary_model(3, 1, k, grad_k, gamma, qT)
x = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))

t0 = time.time()
drift = tv_betatilde(0,None)
t1 = time.time()
print(t1-t0)

t0 = time.time()
diffusion = tv_Btilde(0,None)
t1 = time.time()
print(t1-t0)

t0 = time.time()
tv_diffusion_tilde(0,None)
t1 = time.time()
print(t1-t0)

delta = jnp.array([0.1, 0.2, 0.3])
gamma = jnp.array([0.5])

ahs_drift, ahs_diffusion = lm.ahs_model(3, 1, k, grad_k, k, grad_k, delta, gamma, None)

t0 = time.time()
drift = ahs_drift(0, x, None)
t1 = time.time()
print(t1-t0)

t0 = time.time()
diffusion = ahs_diffusion(0, x, None)
t1 = time.time()
print(t1-t0)


ahs_betatilde, ahs_Btilde, ahs_diffusion_tilde= lm.ahs_auxillary_model(3, 1, k, grad_k, k, grad_k, delta, gamma, None, qT)


t0 = time.time()
beta = ahs_betatilde(0,None)
t1 = time.time()
print(t1-t0)

t0 = time.time()
B = ahs_Btilde(0,None)
t1 = time.time()
print(t1-t0)

t0 = time.time()
diff = ahs_diffusion_tilde(0,None)
t1 = time.time()
print(t1-t0)

