#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 19:13:13 2022

@author: root
"""

#%% Modules

import jax.numpy as jnp

import matplotlib.pyplot as plt

import landmark_models2 as lm
import sim_sp as sp

#%% Functions

#Kernel function
def k(x:jnp.ndarray, y, theta:jnp.ndarray=1.0)->jnp.ndarray:
    
    return jnp.exp(-jnp.dot(x-y,x-y)/(2*(theta**2)))

#Kernel gradient
def grad_k(x:jnp.ndarray, y, theta:jnp.ndarray=1.0)->jnp.ndarray:
    
    return -(theta**(-2))*k(x,y,theta)*(x-y)

#Kernel function
def k_tau(x:jnp.ndarray, y, theta:jnp.ndarray=None)->jnp.ndarray:
    
    theta = 0.5
    
    return jnp.exp(-jnp.dot(x-y,x-y)/(2*(theta**2)))

#Kernel gradient
def grad_k_tau(x:jnp.ndarray, y, theta:jnp.ndarray=None)->jnp.ndarray:
    
    theta = 0.5
    
    return -(theta**(-2))*k_tau(x,y,theta)*(x-y)

#Kernel gradient
def grad_grad_k_tau(x:jnp.ndarray, y, theta:jnp.ndarray=None)->jnp.ndarray:
    
    theta = 0.5
    
    return (theta**(-4))*k_tau(x,y,theta)*((x-y)**2)-(theta**(-2))*k_tau(x,y,theta)

#%% Global variables

n = 3
d = 1
q0 = jnp.array([-0.5, 0.0, 0.1])
p0 = jnp.array([-0.5, -0.2, 0.5])
#p0 = jnp.zeros(3)
qT = jnp.array([-0.5, 0.2, 1.0])

x0 = jnp.hstack((q0,p0))

time_grid = jnp.linspace(0,1,1000)

theta = 1.0

#%% TV-Model 


gamma = 1/jnp.sqrt(n)*jnp.ones(n)

b_fun, sigma_fun = lm.tv_model(n, d, k, grad_k, gamma)
beta_fun, B_fun, sigmatilde_fun = \
    lm.tv_auxillary_model(n, d, k, grad_k, gamma, qT)

b_fun_theta = lambda t,x,theta=theta: b_fun(t,x,theta)
sigma_fun_theta = lambda t,x,theta=theta: sigma_fun(t,x,theta)

beta = beta_fun(0, theta) #Since constant in time
B = B_fun(0, theta) #Since constant in time
sigmatilde = sigmatilde_fun(0, theta) #Since constant in time

beta_fun = lambda t,theta=theta: beta #Since constant in time
B_fun = lambda t,theta=theta: B #Since constant in time
sigmatilde_funfast = lambda t,theta=theta: sigmatilde #Since constant in time
b_funsimple = lambda t,x,theta=theta : b_fun(t,x,theta)
sigma_funsimple = lambda t,x,theta=theta : sigma_fun(t,x,theta)
    
Wt = sp.sim_Wt(time_grid, dim=3, simulations=10)
Xt = jnp.zeros((10, 1000, 6))
for i in range(10):
    Xt = Xt.at[i].set(sp.sim_sde_euler(x0, b_fun_theta, sigma_fun_theta, Wt[i], time_grid))

fig, ax = plt.subplots(1,2, figsize=(8,6))  
ax[0].plot(time_grid, Xt[i,:,0], color='black', label='$q_{1}(t)$')
ax[0].plot(time_grid, Xt[i,:,1], color='blue', label='$q_{2}(t)$')
ax[0].plot(time_grid, Xt[i,:,2], color='red', label='$q_{3}(t)$')

ax[1].plot(time_grid, Xt[i,:,3], color='black', label='$p_{1}(t)$')
ax[1].plot(time_grid, Xt[i,:,4], color='blue', label='$p_{2}(t)$')
ax[1].plot(time_grid, Xt[i,:,5], color='red', label='$p_{3}(t)$')
for i in range(1,10):
    ax[0].plot(time_grid, Xt[i,:,0], color='black')
    ax[0].plot(time_grid, Xt[i,:,1], color='blue')
    ax[0].plot(time_grid, Xt[i,:,2], color='red')
    
    ax[1].plot(time_grid, Xt[i,:,3], color='black')
    ax[1].plot(time_grid, Xt[i,:,4], color='blue')
    ax[1].plot(time_grid, Xt[i,:,5], color='red')
    
fig.suptitle('TV-Model')
ax[0].grid()
ax[0].set_xlabel('$t$')
ax[0].set_title('Landmarks')
ax[0].legend()
ax[1].grid()
ax[1].set_xlabel('$t$')
ax[1].set_title('Momentum')
ax[1].legend()
plt.tight_layout()

btilde_fun = lambda t,x: beta_fun(t)+B_fun(t).dot(x)
Xt = jnp.zeros((10, 1000, 6))
for i in range(10):
    Xt = Xt.at[i].set(sp.sim_sde_euler(x0, btilde_fun, sigmatilde_funfast, Wt[i], time_grid))

fig, ax = plt.subplots(1,2, figsize=(8,6))  
ax[0].plot(time_grid, Xt[i,:,0], color='black', label=r'$\tilde{q}_{1}(t)$')
ax[0].plot(time_grid, Xt[i,:,1], color='blue', label=r'$\tilde{q}_{2}(t)$')
ax[0].plot(time_grid, Xt[i,:,2], color='red', label=r'$\tilde{q}_{3}(t)$')

ax[1].plot(time_grid, Xt[i,:,3], color='black', label=r'$\tilde{p}_{1}(t)$')
ax[1].plot(time_grid, Xt[i,:,4], color='blue', label=r'$\tilde{p}_{2}(t)$')
ax[1].plot(time_grid, Xt[i,:,5], color='red', label=r'$\tilde{p}_{3}(t)$')
for i in range(1,10):
    ax[0].plot(time_grid, Xt[i,:,0], color='black')
    ax[0].plot(time_grid, Xt[i,:,1], color='blue')
    ax[0].plot(time_grid, Xt[i,:,2], color='red')
    
    ax[1].plot(time_grid, Xt[i,:,3], color='black')
    ax[1].plot(time_grid, Xt[i,:,4], color='blue')
    ax[1].plot(time_grid, Xt[i,:,5], color='red')
    
fig.suptitle('TV-Model')
ax[0].grid()
ax[0].set_xlabel('$t$')
ax[0].set_title('Landmarks')
ax[0].legend()
ax[1].grid()
ax[1].set_xlabel('$t$')
ax[1].set_title('Momentum')
ax[1].legend()
plt.tight_layout()

#%% MS-Model 

gamma = 1/jnp.sqrt(n)*jnp.ones(n)
lmbda = 1.0

b_fun, sigma_fun = lm.ms_model(n, d, k, grad_k, lmbda, gamma)
betatilde_fun, Btilde_fun, sigmatilde_fun = \
    lm.ms_auxillary_model(n, d, k, grad_k, lmbda, gamma, qT)

b_fun_theta = lambda t,x,theta=theta: b_fun(t,x,theta)
sigma_fun_theta = lambda t,x,theta=theta: sigma_fun(t,x,theta)

beta = beta_fun(0, theta) #Since constant in time
B = B_fun(0, theta) #Since constant in time
sigmatilde = sigmatilde_fun(0, theta) #Since constant in time

beta_fun = lambda t,theta=theta: beta #Since constant in time
B_fun = lambda t,theta=theta: B #Since constant in time
sigmatilde_funfast = lambda t,theta=theta: sigmatilde #Since constant in time
b_funsimple = lambda t,x,theta=theta : b_fun(t,x,theta)
sigma_funsimple = lambda t,x,theta=theta : sigma_fun(t,x,theta)
    
Wt = sp.sim_Wt(time_grid, dim=3, simulations=10)
Xt = jnp.zeros((10, 1000, 6))
for i in range(10):
    Xt = Xt.at[i].set(sp.sim_sde_euler(x0, b_fun_theta, sigma_fun_theta, Wt[i], time_grid))

fig, ax = plt.subplots(1,2, figsize=(8,6))  
ax[0].plot(time_grid, Xt[i,:,0], color='black', label='$q_{1}(t)$')
ax[0].plot(time_grid, Xt[i,:,1], color='blue', label='$q_{2}(t)$')
ax[0].plot(time_grid, Xt[i,:,2], color='red', label='$q_{3}(t)$')

ax[1].plot(time_grid, Xt[i,:,3], color='black', label='$p_{1}(t)$')
ax[1].plot(time_grid, Xt[i,:,4], color='blue', label='$p_{2}(t)$')
ax[1].plot(time_grid, Xt[i,:,5], color='red', label='$p_{3}(t)$')
for i in range(1,10):
    ax[0].plot(time_grid, Xt[i,:,0], color='black')
    ax[0].plot(time_grid, Xt[i,:,1], color='blue')
    ax[0].plot(time_grid, Xt[i,:,2], color='red')
    
    ax[1].plot(time_grid, Xt[i,:,3], color='black')
    ax[1].plot(time_grid, Xt[i,:,4], color='blue')
    ax[1].plot(time_grid, Xt[i,:,5], color='red')
    
fig.suptitle('MS-Model')
ax[0].grid()
ax[0].set_xlabel('$t$')
ax[0].set_title('Landmarks')
ax[0].legend()
ax[1].grid()
ax[1].set_xlabel('$t$')
ax[1].set_title('Momentum')
ax[1].legend()
plt.tight_layout()

btilde_fun = lambda t,x: beta_fun(t)+B_fun(t).dot(x)
Xt = jnp.zeros((10, 1000, 6))
for i in range(10):
    Xt = Xt.at[i].set(sp.sim_sde_euler(x0, btilde_fun, sigmatilde_funfast, Wt[i], time_grid))

fig, ax = plt.subplots(1,2, figsize=(8,6))  
ax[0].plot(time_grid, Xt[i,:,0], color='black', label=r'$\tilde{q}_{1}(t)$')
ax[0].plot(time_grid, Xt[i,:,1], color='blue', label=r'$\tilde{q}_{2}(t)$')
ax[0].plot(time_grid, Xt[i,:,2], color='red', label=r'$\tilde{q}_{3}(t)$')

ax[1].plot(time_grid, Xt[i,:,3], color='black', label=r'$\tilde{p}_{1}(t)$')
ax[1].plot(time_grid, Xt[i,:,4], color='blue', label=r'$\tilde{p}_{2}(t)$')
ax[1].plot(time_grid, Xt[i,:,5], color='red', label=r'$\tilde{p}_{3}(t)$')
for i in range(1,10):
    ax[0].plot(time_grid, Xt[i,:,0], color='black')
    ax[0].plot(time_grid, Xt[i,:,1], color='blue')
    ax[0].plot(time_grid, Xt[i,:,2], color='red')
    
    ax[1].plot(time_grid, Xt[i,:,3], color='black')
    ax[1].plot(time_grid, Xt[i,:,4], color='blue')
    ax[1].plot(time_grid, Xt[i,:,5], color='red')
    
fig.suptitle('MS-Model')
ax[0].grid()
ax[0].set_xlabel('$t$')
ax[0].set_title('Landmarks')
ax[0].legend()
ax[1].grid()
ax[1].set_xlabel('$t$')
ax[1].set_title('Momentum')
ax[1].legend()
plt.tight_layout()


#%% AHS-Model 

delta = jnp.linspace(-2.5, 2.5, 6)
gamma = jnp.array(0.1)
lmbda = 0.1

b_fun, sigma_fun = lm.ahs_model(n, d, k, grad_k, k_tau, grad_k_tau, grad_grad_k_tau,
                                    delta, gamma)
    
betatilde_fun, Btilde_fun, sigmatilde_fun = lm.ahs_auxillary_model(n, d, k, 
                                                                   grad_k, k_tau, 
                                                                   grad_k_tau,
                                                                   grad_grad_k_tau,
                                                                   delta, gamma, 
                                                                   qT)

b_fun_theta = lambda t,x,theta=theta: b_fun(t,x,theta)
sigma_fun_theta = lambda t,x,theta=theta: sigma_fun(t,x,theta)
beta_fun_theta = lambda t,theta=theta: betatilde_fun(t,theta)
Btilde_fun_theta = lambda t,theta=theta: Btilde_fun(t,theta)
sigmatilde_fun_theta = lambda t,theta=theta: sigmatilde_fun(t,theta)


Wt = sp.sim_Wt(time_grid, dim=6, simulations=10)
Xt = jnp.zeros((10, 1000, 6))
for i in range(10):
    Xt = Xt.at[i].set(sp.sim_sde_euler(x0, b_fun_theta, sigma_fun_theta, Wt[i], time_grid))

fig, ax = plt.subplots(1,2, figsize=(8,6))  
ax[0].plot(time_grid, Xt[i,:,0], color='black', label='$q_{1}(t)$')
ax[0].plot(time_grid, Xt[i,:,1], color='blue', label='$q_{2}(t)$')
ax[0].plot(time_grid, Xt[i,:,2], color='red', label='$q_{3}(t)$')

ax[1].plot(time_grid, Xt[i,:,3], color='black', label='$p_{1}(t)$')
ax[1].plot(time_grid, Xt[i,:,4], color='blue', label='$p_{2}(t)$')
ax[1].plot(time_grid, Xt[i,:,5], color='red', label='$p_{3}(t)$')
for i in range(1,10):
    ax[0].plot(time_grid, Xt[i,:,0], color='black')
    ax[0].plot(time_grid, Xt[i,:,1], color='blue')
    ax[0].plot(time_grid, Xt[i,:,2], color='red')
    
    ax[1].plot(time_grid, Xt[i,:,3], color='black')
    ax[1].plot(time_grid, Xt[i,:,4], color='blue')
    ax[1].plot(time_grid, Xt[i,:,5], color='red')
    
fig.suptitle('AHS-Model')
ax[0].grid()
ax[0].set_xlabel('$t$')
ax[0].set_title('Landmarks')
ax[0].legend()
ax[1].grid()
ax[1].set_xlabel('$t$')
ax[1].set_title('Momentum')
ax[1].legend()
plt.tight_layout()

btilde_fun = lambda t,x: beta_fun_theta(t)+Btilde_fun_theta(t).dot(x)
Xt = jnp.zeros((10, 1000, 6))
for i in range(10):
    Xt = Xt.at[i].set(sp.sim_sde_euler(x0, btilde_fun, sigmatilde_fun_theta, Wt[i], time_grid))

fig, ax = plt.subplots(1,2, figsize=(8,6))  
ax[0].plot(time_grid, Xt[i,:,0], color='black', label=r'$\tilde{q}_{1}(t)$')
ax[0].plot(time_grid, Xt[i,:,1], color='blue', label=r'$\tilde{q}_{2}(t)$')
ax[0].plot(time_grid, Xt[i,:,2], color='red', label=r'$\tilde{q}_{3}(t)$')

ax[1].plot(time_grid, Xt[i,:,3], color='black', label=r'$\tilde{p}_{1}(t)$')
ax[1].plot(time_grid, Xt[i,:,4], color='blue', label=r'$\tilde{p}_{2}(t)$')
ax[1].plot(time_grid, Xt[i,:,5], color='red', label=r'$\tilde{p}_{3}(t)$')
for i in range(1,10):
    ax[0].plot(time_grid, Xt[i,:,0], color='black')
    ax[0].plot(time_grid, Xt[i,:,1], color='blue')
    ax[0].plot(time_grid, Xt[i,:,2], color='red')
    
    ax[1].plot(time_grid, Xt[i,:,3], color='black')
    ax[1].plot(time_grid, Xt[i,:,4], color='blue')
    ax[1].plot(time_grid, Xt[i,:,5], color='red')
    
fig.suptitle('AHS-Model')
ax[0].grid()
ax[0].set_xlabel('$t$')
ax[0].set_title('Landmarks')
ax[0].legend()
ax[1].grid()
ax[1].set_xlabel('$t$')
ax[1].set_title('Momentum')
ax[1].legend()
plt.tight_layout()

