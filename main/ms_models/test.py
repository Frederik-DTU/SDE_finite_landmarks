#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 00:37:16 2022

@author: root
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt


test = jnp.load('simple_saved/Wt_Xt_theta_14500.npz')
Wt = test['Wt']
Xt = test['Xt']

print(Xt[-1])
plt.plot(Xt[:,0:3])