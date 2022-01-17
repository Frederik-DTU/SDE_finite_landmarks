#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:55:05 2021

@author: root
"""
#%% Modules used

import jax.numpy as jnp
from jax import lax, grad
from scipy.optimize import minimize

#%% Functions

def bvp_solver(p0_init, q0, qT, f_fun, grid, max_iter=100, tol=1e-05):
    
    if p0_init is None:
        p0_init = jnp.zeros(0)
    
    idx = len(qT)
    
    def error_fun(p0):
        
        x0 = jnp.hstack((q0,p0))
        
        qT_hat = ode_integrator(x0, f_fun, grid, method='euler')[-1,:idx]
        
        return jnp.sum((qT-qT_hat)**2)
    
    grad_error = grad(error_fun)
    
    sol = minimize(error_fun, p0_init.reshape(-1), jac=grad_error,
                 method='BFGS', options={'gtol': tol, 'maxiter':max_iter, 'disp':True})
    
    print(sol.message)
    
    p0 = sol.x
    x0 = jnp.hstack((q0, p0))
    
    xt = ode_integrator(x0, f_fun, grid, method='euler')
    
    return xt

def ode_integrator(x0, f_fun, grid, method='euler'):
    
    dt_grid = jnp.hstack((jnp.diff(grid)))
    n = jnp.arange(len(dt_grid))
        
    def euler_method():
        
        def step_fun(carry, idx):
            
            t = grid[idx+1]
            dt = dt_grid[idx]
            y = carry+f_fun(t,carry)*dt
            
            return y, y
        
        _, yt = lax.scan(step_fun, x0, xs=n)
        
        return jnp.concatenate((x0[jnp.newaxis,...],yt), axis=0)
    
    return euler_method()

def integrator(f, grid, method='euler'):
    
    dt_grid = jnp.diff(grid)
    n = jnp.arange(len(dt_grid))
    
    if callable(f):
        vec = False
    else:
        vec = True
    
    def euler_method():
        
        def euler_vec():
                        
            dim = jnp.ones(len(f.shape), dtype=int)
            dim = dim.at[0].set(-1)
                        
            val = f[1:]
            zero = jnp.zeros_like(f[0])[jnp.newaxis,...]

            res = jnp.cumsum(jnp.concatenate((zero, \
                                         dt_grid.reshape(dim)*val)),axis=0)
            
            
                
            return res[-1], res
        
        def euler_fun():
            
            def euler_step(carry, idx):
                
                t = grid[idx+1]
                dt = dt_grid[idx]
                val = carry+f(t)*dt
                
                return val, val
                        
            yT, yt = lax.scan(euler_step, 0.0, xs=n)
            
            zero = jnp.zeros_like(yT)[jnp.newaxis,...]
            
            return yT, jnp.concatenate((zero, yt), axis=0)
        
        if vec:
            return euler_vec()
        else:
            return euler_fun()
        
    def trapez_method():
        
        def trapez_vec():
            
            n = len(f.shape)
            if n>1:
                dim = jnp.ones(len(f.shape), dtype=int)
                dim = dim.at[0].set(-1)
            else:
                dim = -1
            
            y_right = f[1:]
            y_left = f[:-1]
            zero = jnp.zeros_like(f[0])[jnp.newaxis,...]
            
            res = jnp.concatenate((zero, \
                                jnp.cumsum(dt_grid.reshape(dim)*(y_right+y_left), axis=0)/2), \
                                  axis=0)
                
            return res[-1], res
            
        def trapez_fun():
            
            def trapez_step(carry, idx):
                
                val_prev, f_prev = carry
                
                t = grid[idx+1]
                dt = dt_grid[idx]
                f_up = f(t)
                val = val_prev+(f_prev+f_up)*dt
                
                return (val, f_up), val
            
            yT, yt = lax.scan(trapez_step, (0.0, f(grid[0])), xs=n)
            
            int_val = yT[0]/2
            zero = jnp.zeros_like(int_val)[jnp.newaxis,...]
            
            return int_val, jnp.concatenate((zero, yt/2), axis=0)
        
        if vec:
            return trapez_vec()
        else:
            return trapez_fun()
                
    
    if method=='trapez':
        return trapez_method()
    else:
        return euler_method()
    