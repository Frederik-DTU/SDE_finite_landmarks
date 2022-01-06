#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 00:40:37 2022

@author: root
"""

#%% Modules

import jax.numpy as jnp

import time

import sim_sp as sp

#%% Testing

t = jnp.linspace(0,1,100)
t0 = time.time()
Wt = sp.sim_Wt(t)
t1 = time.time()
print(t1-t0)