#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 17:26:30 2021

@author: Frederik
"""

#%% Sources

#%% Modules used

import jax.numpy as jnp
from jax import random

from typing import Callable, List

#%% Class

class sde_fun(object):
    def __init__(self, 
                 f:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 g:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 f_args:List=None,
                 g_args:List=None,
                 seed:int = 1071):
        
        self.f = f
        self.g = g
        self.f_args = f_args
        self.g_args = g_args
        self.key = random.PRNGKey(seed)
        
    def sim_Wt(self, shape:List[int]=[10,100,1], tl:List[float]=[0.,1.]):
        
        t0 = tl[0]
        T = tl[1]
        dt = (T-t0)/shape[1]
        Wt = jnp.zeros(shape)
        Wt[:,1:,:] = Wt.set[:,1:,:].set(jnp.sqrt(dt)*
                                        random.normal(self.key,
                                                      [shape[0], shape[1] ,shape[3]]))
        
        return jnp.cumsum(Wt, axis=0)
    
    def sim_sde(self, y0:jnp.ndarray, n_sim:int = 10, n_steps:int=100, tl:List=[0,1]):
        
        t0 = tl[0]
        T = tl[1]
        dt = (T-t0)/n_steps
        shape = [n_sim, n_steps]+list(y0.shape)
        sim = jnp.zeros(shape)

        Wt = jnp.sqrt(dt)*random.normal(self.key, [n_sim, n_steps-1, y0.shape[0]])
        for i in range(n_sim):
            print("Computing simulation: ", i+1)
            sim = sim.at[i,0,:].set(y0)
            for j in range(1,n_steps):
                t = t0+dt*j
                sim = sim.at[i,j,:].set(sim[i,j-1,:]+
                                        self.f(t, sim[i,j-1,:],self.f_args)*dt+
                                           self.g(t, sim[i,j-1,:],self.g_args)*Wt[i,j-1,:])
                
        t = jnp.linspace(t0,T,n_steps)
        
        return t, sim
    
    def ito_integral(self, Gt:jnp.ndarray, #dim(Gt) = [sim, steps, shape]
                 n_sim:int=10, n_Steps:int=100, tl:List[float]=[0.,1.]):
        
        n_sim = Gt.shape[0]
        n_steps = Gt.shape[1]
        
        ito_int = jnp.zeros(n_sim, Gt.shape[0])
        Wt = self.sim_Wt(Gt.shape, tl)
        
        for n in range(n_sim):
            for i in range(1,n_steps):
                ito_int += ito_int.at[n,:].set(Gt[n,i-1,:]*
                                                   (Wt[n,i,:]-Wt[n,i,:]))
                
        return ito_int
    
    def stratonovich_integral(self, Gt:jnp.ndarray, #dim(Gt) = [sim, steps, shape]
                 n_sim:int=10, n_Steps:int=100, tl:List[float]=[0.,1.]):
        
        n_sim = Gt.shape[0]
        n_steps = Gt.shape[1]
        
        strat_int = jnp.zeros(n_sim, Gt.shape[0])
        Wt = self.sim_Wt(Gt.shape, tl)
        
        for n in range(n_sim):
            for i in range(1,n_steps):
                strat_int += strat_int.at[n,:].set((Gt[n,i-1,:]+Gt[n,i,:])*
                                                   (Wt[n,i,:]-Wt[n,i,:]))
                
        return 1/2*strat_int