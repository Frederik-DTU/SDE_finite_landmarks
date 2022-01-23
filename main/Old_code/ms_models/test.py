#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 00:37:16 2022

@author: root
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt

time_grid = jnp.arange(0,1+0.001, 0.001)
time_grid = time_grid*(2-time_grid)
N = int(20000/500)
Xt = jnp.zeros((N,1001,6))
for i in range(1,N):
    test = jnp.load('simple_saved/Wt_Xt_'+str(i*500)+'.npz')
    Xt = Xt.at[i].set(test['Xt'])
    
plt.figure(figsize=(8,6))
alpha = jnp.linspace(1/40,1,40)
for i in range(N):
    plt.plot(time_grid, Xt[i,:,0], color='black', alpha=float(alpha[i]))
    plt.plot(time_grid, Xt[i,:,1], color='blue', alpha=float(alpha[i]))
    plt.plot(time_grid, Xt[i,:,2], color='red', alpha=float(alpha[i]))
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$X_t$')
plt.title('MS-Model')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 00:37:16 2022

@author: root
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt

time_grid = jnp.arange(0,1+0.01, 0.01)
time_grid = time_grid*(2-time_grid)
N = int(20000/500)
Xt = jnp.zeros((N,101,6))
for i in range(1,N):
    test = jnp.load('simple_saved/Wt_Xt_theta_'+str(i*500)+'.npz')
    Xt = Xt.at[i].set(test['Xt'])
    
plt.figure(figsize=(8,6))
alpha = jnp.linspace(1/40,1,40)
for i in range(N):
    plt.plot(time_grid, Xt[i,:,0], color='black', alpha=float(alpha[i]))
    plt.plot(time_grid, Xt[i,:,1], color='blue', alpha=float(alpha[i]))
    plt.plot(time_grid, Xt[i,:,2], color='red', alpha=float(alpha[i]))
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$X_t$')
plt.title(r'MS-Model with unkown $\theta$')





import jax.numpy as jnp
import matplotlib.pyplot as plt

time_grid = jnp.arange(0,1+0.001, 0.001)
time_grid = time_grid*(2-time_grid)
N = int(20000/500)
Xt = jnp.zeros((N,1001,6))
for i in range(1,N):
    test = jnp.load('simple_template/Wt_Xt_'+str(i*500)+'.npz')
    Xt = Xt.at[i].set(test['Xt'][0])
    
plt.figure(figsize=(8,6))
alpha = jnp.linspace(1/40,1,40)
for i in range(N):
    plt.plot(time_grid, Xt[i,:,0], color='black', alpha=float(alpha[i]))
    plt.plot(time_grid, Xt[i,:,1], color='blue', alpha=float(alpha[i]))
    plt.plot(time_grid, Xt[i,:,2], color='red', alpha=float(alpha[i]))
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$X_t$')
plt.title(r'Template MS-Model')

import jax.numpy as jnp
import matplotlib.pyplot as plt

time_grid = jnp.arange(0,1+0.001, 0.001)
time_grid = time_grid*(2-time_grid)
N = int(17500/500)
Xt = jnp.zeros((N,1001,6))
for i in range(1,N):
    test = jnp.load('simple_template/Wt_Xt_theta'+str(i*500)+'.npz')
    Xt = Xt.at[i].set(test['Xt'][0])
    
plt.figure(figsize=(8,6))
alpha = jnp.linspace(1/40,1,40)
for i in range(N):
    plt.plot(time_grid, Xt[i,:,0], color='black', alpha=float(alpha[i]))
    plt.plot(time_grid, Xt[i,:,1], color='blue', alpha=float(alpha[i]))
    plt.plot(time_grid, Xt[i,:,2], color='red', alpha=float(alpha[i]))
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$X_t$')
plt.title(r'Template MS-Model with unkown $\theta$')

