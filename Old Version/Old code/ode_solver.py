#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:39:43 2021

@author: root
"""

#%% Sources:
"""
Minimize documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
Odeint:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
"""


#%% Modules used

#JAX
import jax.numpy as jnp
from jax import grad

#Optimization
from scipy.optimize import minimize

#ODE Solvers
from scipy.integrate import solve_ivp

#Typing
from typing import Callable, List

#%% Class to solve ODEs

class ode_solver(object):
    def __init__(self, 
                 fun:Callable[[jnp.ndarray], jnp.ndarray],
                 tl:List[float] = [0.,1.],
                 n_steps:int=100,
                 max_iter:int=100):
        
        self.fun = fun
        self.t0 = tl[0]
        self.T =tl[1]
        self.n_steps = 100
        self.dt = (tl[1]-tl[0])/n_steps
        self.max_iter=max_iter
        self.y0 = None
        self.yT = None
        self.dim = None
        
    def eulers_method(self, y0:jnp.ndarray):
        
        y = jnp.zeros([len(y0), self.n_steps])
        y = y.at[:,0].set(y0)
        t = jnp.linspace(self.t0, self.T, self.n_steps)
        
        for i in range(0,self.n_steps-1):
            y = y.at[:,i+1].set(y[:,i]+self.fun(t[i],y[:,i])*self.dt)
            
        return t, y
    
    def __eulers_bvp(self, y0:jnp.ndarray):

        y = y0
        t = jnp.linspace(self.t0, self.T, self.n_steps)
        
        for i in range(0,self.n_steps-1):
            y = y+self.fun(t[i],y)*self.dt
            
        return y
    
    def __bvp_loss(self, y0:jnp.ndarray):
        
        y0_init = jnp.concatenate((self.y0, y0[self.dim:]))
        
        #_, yT = self.solve_ivp(y0_init)
        #yT_guess = yT[0:self.dim,-1].reshape(-1)
        yT_guess = self.__eulers_bvp(y0_init)[0:self.dim]
        
        val = jnp.sum((yT_guess-self.yT)**2)
                
        print(val)

        return val
    
    def bvp_bfgs(self, y0_init:jnp.ndarray,
                 yT:jnp.ndarray,
                 y0_fixed:jnp.ndarray):
        
        self.dim = y0_fixed.shape[0]
        self.y0 = y0_fixed
        self.yT = yT
        grad_bvp = grad(self.__bvp_loss)
        
        res = minimize(self.__bvp_loss, y0_init, jac=grad_bvp,
                       method='BFGS')
        
        return res
    
    def solve_ivp(self, y0:jnp.ndarray, n_grid:int=10, T:float=1)->jnp.ndarray:
        
        x_mesh = jnp.linspace(0, T, n_grid)
        
        sol = solve_ivp(self.fun, [0,T], y0, t_eval = x_mesh)
        
        if sol.status != 0:
            print("WARNING: sol.status is %d" % sol.status)
        
        return sol.t, sol.y
        
        
        
        
        
        
        
        
        
        
        