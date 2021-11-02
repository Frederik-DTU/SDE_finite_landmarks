#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:11 2021

@author: root
"""

#%% Sources:
    
"""
Article: "Diffusion Bridges For Stochastic Hamiltonian Systems and Shape Evolutions"
-https://arxiv.org/abs/2002.00885
JAX Documentation
-https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
"""

#%% Modules

#JAX
import jax.numpy as jnp #Only works on Linux/mac
from jax import grad #Only works on Linux/mac

#Typing
from typing import Callable

#%% Landmark Class

class landmark_models(object):
    def __init__(self, 
                 k:Callable[[jnp.ndarray], jnp.ndarray],
                 dk:Callable[[jnp.ndarray], jnp.ndarray]=None):
        
        self.k = k #Scalar kernel
        if dk is None:
            self.dk = grad(k) #gradient of kernel
        else:
            self.dk = dk
        
    def ham_form(self, y:jnp.ndarray)->jnp.ndarray:
        
        n = int(y.shape[0]/2)
        
        q = y[0:n]
        p = y[n:]
        
        sum_k = 0.0
        for i in range(n):
            for j in range(n):
                sum_k += jnp.dot(p[i], p[j])*self.k(q[i]-q[j])
                
        return 1/2*sum_k
    
    def dqi(self, qi:jnp.ndarray, q:jnp.ndarray, p:jnp.ndarray)->jnp.ndarray:
        
        n = q.shape[0]
        dqi = jnp.zeros_like(qi)
        
        for j in range(n):
            dqi += p[j]*self.k(qi-q[j])
            
        return dqi
    
    def dpi(self, pi:jnp.ndarray, qi:jnp.ndarray, q:jnp.ndarray, p:jnp.ndarray)->jnp.ndarray:
        
        n = q.shape[0]
        dpi = jnp.zeros_like(pi)
        
        for j in range(n):
            dpi += jnp.dot(pi,p[j])*self.dk(qi-q[j])
            
        return -dpi
    
    def tv_drift(self, t:jnp.ndarray, y:jnp.ndarray, *args)->jnp.ndarray:
        
        n = int(y.shape[0]/2)
        q = y[0:n]
        p = y[n:]
        dq = jnp.zeros_like(q)
        dp = jnp.zeros_like(p)
        
        for i in range(n):
            dq = dq.at[i].set(self.dqi(q[i], q, p))
            dp = dp.at[i].set(self.dpi(p[i], q[i], q, p))
            
        return jnp.concatenate((dq, dp))
    
    def tv_diffusion(self, t:jnp.ndarray,
                     y:jnp.ndarray, *args)->jnp.ndarray:
        
        n = int(y.shape[0]/2)
        if len(args) == 0:
            gamma = jnp.ones(n)
        else:
            gamma = args[0]
        
        dif = jnp.zeros(int(2*n))
        dif = dif.at[n:].set(gamma)
        
        return dif
    
    def ms_drift(self, t:jnp.ndarray, y:jnp.ndarray, *args)->jnp.ndarray:
        
        if len(args)==0:
            lam = 1
        else:
            lam = args[0]
        
        n = int(y.shape[0]/2)
        q = y[0:n]
        p = y[n:]
        dq = jnp.zeros_like(q)
        dp = jnp.zeros_like(p)
        
        for i in range(n):
            dq = dq.at[i].set(self.dqi(q[i], q, p))
            dp = dp.at[i].set(-lam*dq[i]-self.dpi(p[i], q[i], q, p))
            
        return jnp.concatenate((dq, dp))
    
    def ms_diffusion(self, t:jnp.ndarray,
                     y:jnp.ndarray, *args)->jnp.ndarray:
        
        n = int(y.shape[0]/2)
        if len(args) == 0:
            gamma = jnp.ones(n)
        else:
            gamma = args[0]
        
        dif = jnp.zeros(int(2*n))
        dif = dif.at[n:].set(gamma)
        
        return dif
    
    def ahs_drift(self, y:jnp.ndarray, gamma:jnp.ndarray, 
                  delta:jnp.ndarray,
                  k_tau:Callable[[jnp.ndarray], jnp.ndarray])->jnp.ndarray:
        
        t = jnp.array([0.0])
        drift_strat = self.tv_drift(t, y)
        n = int(y.shape[0]/2)
        J = len(delta.shape[0])
        q = y[0:n]
        p = y[n:]
        dq = jnp.zeros_like(q)
        dp = jnp.zeros_like(p)
        grad_k_tau = grad(k_tau)
                
        for i in range(n):
            for l in range(J):
                zl = lambda qi: jnp.dot(grad_k_tau(qi-delta[l]),gamma[l])
                grad_zl = grad(zl)
                dq[i] += 1/2*(jnp.dot(grad_k_tau(q[i]-delta[l]), gamma[l])*
                              k_tau(q[i]-delta[l])*gamma[l])
                dp[i] += 1/2*(jnp.dot(p[i], gamma[l])*(zl(q[i])*grad_k_tau(q[i]-delta[l])-
                              k_tau(q[i]-delta[l])*grad_zl(q[i])))
                
        extra_drift = jnp.concatenate((dq, dp))
        
        drift = drift_strat+extra_drift
        
        return drift
    
    def ahs_diffusion(self, y:jnp.ndarray, gamma:jnp.ndarray,
                      delta:jnp.ndarray,
                      k_tau:Callable[[jnp.ndarray], jnp.ndarray])->jnp.ndarray:
        
        n = int(y.shape[0]/2)
        J = len(delta.shape[0])
        q = y[0:n]
        p = y[n:]
        dq = jnp.zeros_like(q)
        dp = jnp.zeros_like(p)
        
        for i in range(n):
            for j in range(J):
                sigma_l = lambda qi: p[i]*gamma[i]*k_tau(qi-delta[j])
                grad_sigma = grad(sigma_l)
                dq[i] += gamma[i]*k_tau(q[i]-delta[j])
                dp[i] -= grad_sigma(q[i])
        
        return jnp.concatenate((dq, dp))
        
            
        
        
        