#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:30:51 2021

@author: root
"""

#%% Modules

#JAX
import jax.numpy as jnp

#Plotting
import matplotlib.pyplot as plt

#Own modules
from sde_approx import sde_fun
from plot_landmarks import plot_landmarks

#%% Functions used

def k(x):
    return jnp.exp(-jnp.dot(x,x)/2)

#%% Deterministic 1d results

N = 10
q0 = jnp.linspace(0, 1, N).reshape(-1,1)
p0 = jnp.linspace(-1,1, N).reshape(-1,1)

sde = sde_fun()
plot_land = plot_landmarks()

x0 = jnp.hstack((q0,p0))
t_ivp, qt_ivp, pt_ivp = sde.landmark_shooting_ivp_rk45(x0, k)
qT = qt_ivp[-1].reshape(-1,1)
plot_land.plot_1d_landmarks_ivp(t_ivp, qt_ivp, pt_ivp)

t_bvp, qt_bvp, pt_bvp = sde.landmark_matching_bfgs(q0, qT, k)
plot_land.plot_1d_landmarks_bvp(t_bvp, qt_bvp, pt_bvp)

#%% TV-model 1d results

q0 = jnp.array([-0.5, 0.0, 0.1]).reshape(-1,1)
p0 = jnp.array([-1., 0., 1.]).reshape(-1,1)
gamma = jnp.array([0.5, 0.2, 0.1])

t_tv, qt_tv, pt_tv = sde.sim_tv_model(q0, p0, k, gamma)
qt_tv = qt_tv.squeeze()
pt_tv = pt_tv.squeeze()

plot_land.plot_1d_landmarks_realisations(t_tv, qt_tv, pt_tv)

#%% MS-model 1d results

q0 = jnp.array([-0.5, 0.0, 0.1]).reshape(-1,1)
p0 = jnp.array([-1., 0., 1.]).reshape(-1,1)
theta = jnp.array([0.5, 0.5, 0.2, 0.1])

t_ms, qt_ms, pt_ms = sde.sim_ms_model(q0, p0, k, theta)
qt_ms = qt_ms.squeeze()
pt_ms = pt_ms.squeeze()

plot_land.plot_1d_landmarks_realisations(t_ms, qt_ms, pt_ms)


