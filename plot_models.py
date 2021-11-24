#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 02:25:45 2021

@author: root
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

file_path = 'Model_output/1d_landmarks_5256.npy'
Xt = jnp.load(file_path)

plt.figure()
plt.plot(Xt)

file_path = 'Model_output/1d_landmarks_5578.npy'
Xt = jnp.load(file_path)

plt.figure()
plt.plot(Xt)