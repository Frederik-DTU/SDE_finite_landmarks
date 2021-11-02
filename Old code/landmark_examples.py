#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 17:30:48 2021

@author: root
"""

#%% Sources:
    
"""

"""

#%% Modules

#JAX
import jax.numpy as jnp

#Plotting
import matplotlib.pyplot as plt

#Own modules
from landmark_models import landmark_models 
from ode_solver import ode_solver
from sde_fun import sde_fun

#%% Kernels

def gaussian_kernel(x:jnp.ndarray)->jnp.ndarray:
    
    return jnp.exp(-jnp.dot(x,x)/2)

#%% Deterministic model

lm = landmark_models(gaussian_kernel)
ode = ode_solver(lm.tv_drift)

q0 = jnp.array([-0.5, 0.0, 0.1]) #Initial landmarks
p0 = jnp.linspace(-1,1,3) #Initial moment
y0 = jnp.concatenate((q0, p0)) #concatenate
t, y = ode.solve_ivp(y0) #Solutions

qt = y[0:len(q0)] #Computed landmarks
pt = y[len(q0):] #Computed moments

#Plotting qt
plt.figure(figsize=((8,6)))
plt.plot(t, qt.transpose())
plt.title("q")
plt.ylabel("Position")
plt.xlabel("t")
plt.legend(["q1(t)", "q2(t)", "q3(t)"])
plt.grid()

#Plotting pt
plt.figure(figsize=((8,6)))
plt.plot(t, pt.transpose())
plt.title("p")
plt.ylabel("Momentum")
plt.xlabel("t")
plt.legend(["p1(t)", "p2(t)", "p3(t)"])
plt.grid()

#Solving with given end point
ode = ode_solver(lm.tv_drift)
q0 = jnp.array([-0.5, 0.0, 0.1]) #Initial landmarks
qT = jnp.array([-0.5, 0.2, 1.0]) #End point
p0_init = jnp.array([-1.,-0,1])
y0_init = jnp.concatenate((q0, p0_init))
#res = ode.bvp_bfgs(y0_init, qT, q0) #Takes some time

#%% Realisations of TV

lm = landmark_models(gaussian_kernel)
q0 = jnp.array([-0.5, 0.0, 0.1]) #Initial landmarks
p0 = jnp.linspace(-1,1,3) #Initial moment
y0 = jnp.concatenate((q0, p0)) #concatenate

g_args = jnp.array([1.,1.,1.])
sde = sde_fun(lm.tv_drift, lm.tv_diffusion, g_args = g_args)
t, sim = sde.sim_sde(y0)

qt = sim[:,:,0:len(q0)] #Computed landmarks
pt = sim[:,:,len(q0):] #Computed moments

#Plotting qt

fig, ax = plt.subplots(len(q0), 2, figsize=((8,6)))
ax[0,0].set_title('q(t)')
ax[0,1].set_title('p(t)')
ax[-1,0].set_xlabel("t")
ax[-1,1].set_xlabel("t")
for i in range(len(q0)):
    ax[i,0].plot(t, qt[:,:,i].transpose())
    ax[i,0].set_ylabel('q'+str(i+1)+'(t)')
    ax[i,0].grid()
    
    ax[i,1].plot(t, pt[:,:,i].transpose())
    ax[i,1].set_ylabel("p"+str(i+1)+'(t)')
    ax[i,1].grid()
    
#%% Realisations of MS

lm = landmark_models(gaussian_kernel)
q0 = jnp.array([-0.5, 0.0, 0.1]) #Initial landmarks
p0 = jnp.linspace(-1,1,3) #Initial moment
y0 = jnp.concatenate((q0, p0)) #concatenate

g_args = jnp.array([1.,1.,1.])
f_args = 1
sde = sde_fun(lm.ms_drift, lm.ms_diffusion, f_args = f_args, g_args = g_args)
t, sim = sde.sim_sde(y0)

qt = sim[:,:,0:len(q0)] #Computed landmarks
pt = sim[:,:,len(q0):] #Computed moments

#Plotting qt

fig, ax = plt.subplots(len(q0), 2, figsize=((8,6)))
ax[0,0].set_title('q(t)')
ax[0,1].set_title('p(t)')
ax[-1,0].set_xlabel("t")
ax[-1,1].set_xlabel("t")
for i in range(len(q0)):
    ax[i,0].plot(t, qt[:,:,i].transpose())
    ax[i,0].set_ylabel('q'+str(i+1)+'(t)')
    ax[i,0].grid()
    
    ax[i,1].plot(t, pt[:,:,i].transpose())
    ax[i,1].set_ylabel("p"+str(i+1)+'(t)')
    ax[i,1].grid()
    
#%% Realisations of AHS

