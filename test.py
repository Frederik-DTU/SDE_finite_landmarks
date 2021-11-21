#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:35:06 2021

@author: root
"""

#%% Testing

sde = sde_finite_landmarks(seed=900)

def b_fun(t, x, theta):
    
    return jnp.ones(4)

def sigma_fun(t, x, theta):
    
    return jnp.eye(4)

def sigmatilde_fun(t, theta):
    
    return jnp.eye(4)

def betatilde_fun(t, theta):
    
    return jnp.zeros(4)

def Btilde_fun(t, theta):
    
    return 2.0*jnp.eye(4)

def pi_prob(p0):
    return multivariate_normal.pdf(p0, mean=jnp.zeros_like(p0),
                                   cov=100*jnp.eye(len(p0)))

def q_theta(theta_circ, theta):
    
    return multivariate_normal.pdf(jnp.log(theta_circ), mean=jnp.log(theta),
                                   cov=jnp.eye(len(theta)))

def q_sample_theta(theta_circ, theta):
    
    theta_circ = sde.sim_multi_normal(mean=jnp.log(theta),cov=jnp.eye(len(theta)))
    
    return theta_circ

p0 = jnp.array([0.0, 0.0]).reshape(-1,1)
q0 = jnp.array([1.0, 1.0]).reshape(-1,1)
qT = jnp.array([1.5, 1.5])
epsilon=1
SigmaT = epsilon**2*jnp.eye(2)
LT = jnp.hstack((jnp.eye(2), jnp.zeros((2,2))))
x0 = jnp.vstack((q0, p0))

vT = qT

val = sde.approx_landmark_sde(q0, p0, vT, SigmaT, LT, b_fun, sigma_fun, 
                              betatilde_fun, Btilde_fun, sigmatilde_fun, 
                              pi_prob, q_theta, q_sample_theta,
                              max_iter=10,
                              n_steps=100)