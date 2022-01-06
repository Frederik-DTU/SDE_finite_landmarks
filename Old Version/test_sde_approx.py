#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:35:06 2021

@author: root
"""

#%% Sources:
    
    

#%% Modules

#JAX
import jax.numpy as jnp

#Plotting
import matplotlib.pyplot as plt

#From scipy
from scipy.stats import multivariate_normal

#Own modules
from sde_approx import sde_finite_landmarks

#%% Testing simple functions

#Consider the SDE
#dXt = mu*Xtdt + sigma*Xt*dWt
#Assume that Xt \in R^4 and that sigma=eye(4) and mu=0.5

sde = sde_finite_landmarks(seed=2712)

#Simulating and plotting Wt
t, Wt = sde.sim_Wt(n_sim=10, dim=1)
plt.figure(figsize=(8,6))
plt.plot(t,Wt.T)

t, Wt = sde.sim_Wt(n_sim=1, dim=4)
plt.figure(figsize=(8,6))
plt.plot(t,Wt)

#Simulating and plotting dWt
t, dWt = sde.sim_dWt(n_sim=10, dim=1)
plt.figure(figsize=(8,6))
plt.plot(t,dWt.T)

t, dWt = sde.sim_dWt(n_sim=1, dim=4)
plt.figure(figsize=(8,6))
plt.plot(t,dWt)

#Testing multivariate normal density
val = sde.multi_normal_pdf(jnp.zeros(2))
print("Code: ", val)
print("Correct: ", multivariate_normal.pdf(jnp.zeros(2), jnp.zeros(2), jnp.eye(2)))

#%% Simulating multivariate geometry Brownian motion

x0 = jnp.array([0.5, 7.5, 10, 2.0])

def b_fun(t, x, theta):
    
    return 1*x

def sigma_fun(t, x, theta):
    
    return jnp.eye(4)*x

t = jnp.linspace(0,1,100)
t, sim = sde.sim_sde(x0, b_fun, sigma_fun, n_sim=1, grid=t)
plt.figure(figsize=(8,6))
plt.plot(t,sim)

#%% Testing integral computatations

def f(t):
    
    return t**2

t = jnp.linspace(0,1,100)
ft = f(t)
ri = sde.ri_trapez(ft, t)
print("The estimated RI integral:", ri)
print("The true RI integral:", 1/3)

print("Ito integral for realisation: ", sde.ito_integral(sim, n_sim_int=10, grid=t))
print("Stratonovich integral for realisation: ", sde.stratonovich_integral(sim, n_sim_int=10, grid=t))

#%% Testing the approximating SDE 

p0 = jnp.array([0.0, 0.0]).reshape(-1,1)
q0 = jnp.array([1.0, 1.0]).reshape(-1,1)
qT = jnp.array([1.5, 1.5])

def b_fun(t, x, theta):
    
    return 1*x

def sigma_fun(t, x, theta):
    
    return jnp.eye(4)*x

def betatilde_fun(t, theta):
    
    return jnp.zeros(4)
    
def Btilde_fun(t, theta):
    
    return 2*jnp.eye(4)
    
def sigmatilde_fun(t, theta):
    
    return jnp.eye(4)

#Kernel for prior
def pi_kernel(x:jnp.ndarray)->jnp.ndarray:
    
    theta = 1.0
    
    return jnp.exp(-jnp.dot(x,x)/((2*theta)**2))

#Prior on landmarks
def pi_x0(q0:jnp.ndarray, p0:jnp.ndarray)->jnp.ndarray:
    
    kmom = 100
    
    Sigma = kmom/jnp.apply_along_axis(pi_kernel, -1, q0)
    mu = jnp.zeros_like(p0.reshape(-1))
        
    pi = multivariate_normal.pdf(p0.reshape(-1), mean=mu,
                                          cov = Sigma)
    
    return pi

epsilon=1
SigmaT = epsilon**2*jnp.eye(2)
LT = jnp.hstack((jnp.eye(2), jnp.zeros((2,2))))
x0 = jnp.vstack((q0, p0))
vT = qT

_, Xt = sde.approx_p0(q0, p0, vT, SigmaT, LT, 
                        b_fun, sigma_fun, 
                        betatilde_fun, Btilde_fun, sigmatilde_fun, 
                        pi_x0,
                        time_grid = t,
                        max_iter = 10,
                        eta = 1.0,
                        delta = 1.0,
                        save_path='',
                        save_hours=10)

