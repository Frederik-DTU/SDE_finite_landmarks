#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:55:05 2021

@author: root
"""
#%% Modules used

import jax.numpy as jnp
from jax import vmap, grad

#%% Functions

def geodesic_eqrhs(n, d, k, grad_k): 
    
    def rhs(t,x):
    
        dim = [2*n,d]
        
        x = x.reshape(dim)
        q = x[0:n]
        p = x[n:]
        
        dq = dH_dp(q, p, k, None).reshape(-1)
        dp = -(dH_dq(q, p, grad_k, None)).reshape(-1)
                
        return jnp.hstack((dq, dp))

    return rhs

def tv_model(n, d, k, grad_k, gamma): 
    
    dim = [2*n,d]
    
    def tv_drift(t, x, theta)->jnp.ndarray:
        
        x = x.reshape(dim)
        q = x[0:n]
        p = x[n:]
        
        dq = dH_dp(q, p, k, theta).reshape(-1)
        dp = -(dH_dq(q, p, grad_k, theta)).reshape(-1)
                
        return jnp.hstack((dq, dp))
    
    def tv_diffusion(t, x, theta):
        
        val = jnp.diag(gamma)
        zero = jnp.zeros_like(val)
                
        return jnp.vstack((zero, val))

    return tv_drift, tv_diffusion
    
def tv_auxillary_model(n, d, k, grad_k, gamma, qT=None):
    
    def tv_betatilde(t,qT, theta):
        
        return jnp.zeros(2*n*d)
    
    def tv_Btilde(t,qT, theta):
        
        def k_qTi(qTi:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
        
            K_val = jnp.zeros((d, n*d))
            K = vmap(lambda x: k(qTi,x,theta))(qT.reshape(n,d)).reshape(-1)
    
            for i in range(d):
                K_val = K_val.at[i,i::d].set(K)
    
            return K_val
        
        K = vmap(lambda x: k_qTi(x,theta))(qT.reshape(n,d)).reshape(-1,n*d)
        
        zero = jnp.zeros_like(K)
        
        K = jnp.vstack((K, zero))
                        
        return jnp.hstack((jnp.zeros_like(K), K))
    
    def tv_diffusion_tilde(t, qT, theta):
        
        val = jnp.diag(gamma)
        zero = jnp.zeros_like(val)
                
        return jnp.vstack((zero, val))
    
    if qT is None:
        beta = tv_betatilde
        B = tv_Btilde
        sigmatilde = tv_diffusion_tilde
    else:
        beta = lambda t,theta: tv_betatilde(t,qT,theta)
        B = lambda t,theta: tv_Btilde(t,qT.reshape(n,d), theta)
        sigmatilde = lambda t,theta: tv_diffusion_tilde(t,qT,theta)
        
    return beta, B, sigmatilde

def ms_model(n, d, k, grad_k, lmbda, gamma): 
    
    dim = [2*n,d]
    
    def ms_drift(t, x, theta)->jnp.ndarray:
        
        x = x.reshape(dim)
        q = x[0:n]
        p = x[n:]
        
        dq = dH_dp(q, p, k, theta).reshape(-1)
        
        dp = -(lmbda*dq+(dH_dq(q, p, grad_k, theta)).reshape(-1))
                
        return jnp.hstack((dq, dp))
    
    def ms_diffusion(t, x, theta):
        
        val = jnp.diag(gamma)
        zero = jnp.zeros_like(val)
                
        return jnp.vstack((zero, val))
    
    return ms_drift, ms_diffusion
    
def ms_auxillary_model(n, d, k, grad_k, lmbda, gamma, qT=None):
    
    def ms_betatilde(t, qT, theta):
            
            return jnp.zeros(2*n*d)
        
    def ms_Btilde(t, qT, theta):
        
        def k_qTi(qTi:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
        
            K_val = jnp.zeros((d, n*d))
            K = vmap(lambda x: k(qTi,x,theta))(qT.reshape(n,d)).reshape(-1)
    
            for i in range(d):
                K_val = K_val.at[i,i::d].set(K)
    
            return K_val
        
        K = vmap(lambda x: k_qTi(x,theta))(qT.reshape(n,d)).reshape(-1,n*d)
        
        K = jnp.vstack((K, -lmbda*K))
        zero = jnp.zeros_like(K)
                        
        return jnp.hstack((zero, K))
    
    def ms_diffusion_tilde(t, qT, theta):
        
        val = jnp.diag(gamma)
        zero = jnp.zeros_like(val)
                
        return jnp.vstack((zero, val))
    
    if qT is None:
        beta = ms_betatilde
        B = ms_Btilde
        sigmatilde = ms_diffusion_tilde
    else:
        beta = lambda t,theta: ms_betatilde(t,qT,theta)
        B = lambda t,theta: ms_Btilde(t,qT, theta)
        sigmatilde = lambda t,theta: ms_diffusion_tilde(t,qT,theta)
        
    return beta, B, sigmatilde

def ahs_model(n, d, k, grad_k, k_tau, grad_k_tau, grad_grad_k_tau, delta, gamma):
    
    def z(qi, delta_l, theta):
        
        return jnp.dot(grad_k_tau(qi,delta_l,theta),gamma).reshape(())
    
    def grad_z(qi, delta_l, theta):
        
        return jnp.dot(grad_grad_k_tau(qi,delta_l,theta),gamma).reshape(())
    
    def sigma_l(qi, delta_l, theta):
        
        return gamma*k_tau(qi,delta_l, theta)
    
    #grad_z = grad(z, argnums=0)
    dim = [2*n, d]
    
    def dqi_extra(qi, theta):
        
        z_val = vmap(lambda delta_l: z(qi, delta_l, theta))(delta)
        k_tau_val = vmap(lambda delta_l: k_tau(qi,delta_l, theta))(delta)
        
        return jnp.dot(z_val,k_tau_val)*gamma
    
    def dpi_extra(qi, pi, theta):
        
        z_val = vmap(lambda delta_l: z(qi, delta_l, theta))(delta)
        grad_k_tau_val = vmap(lambda delta_l: grad_k_tau(qi,delta_l, theta))(delta)
        k_tau_val = vmap(lambda delta_l: k_tau(qi,delta_l, theta))(delta)
        grad_z_val = vmap(lambda delta_l: grad_z(qi, delta_l, theta))(delta)
                
        return (jnp.dot(z_val, grad_k_tau_val)-\
                jnp.dot(k_tau_val, grad_z_val))*jnp.dot(pi,gamma)
        
    def dqi_diffusion(qi, theta):
        
        delta_fun = lambda delta_l: sigma_l(qi, delta_l, theta)
        
        return vmap(delta_fun)(delta).T
    
    def dpi_diffusion(qi, pi, theta):
        
        delta_fun = lambda delta_l: jnp.dot(grad_k_tau(qi,delta_l, theta), pi)*gamma
        
        return vmap(delta_fun)(delta).T
            
    def ahs_drift(t, x, theta):
        
        x = x.reshape(dim)
        q = x[0:n].reshape(n,d)
        p = x[n:].reshape(n,d)
        
        dq = dH_dp(q, p, k, theta).reshape(-1)+1/2*vmap(lambda x: dqi_extra(x,theta))(q).reshape(-1)
        dp = (-dH_dq(q, p, grad_k, theta)+\
              1/2*vmap(lambda x,y: dpi_extra(x,y,theta))(q, p)).reshape(-1)
                
        return jnp.hstack((dq, dp))
    
    def ahs_diffusion(t, x, theta):
        
        x = x.reshape(dim)
        q = x[0:n].reshape(n,d)
        p = x[n:].reshape(n,d)
        
        dq_diffusion = vmap(lambda x: dqi_diffusion(x,theta))(q).reshape(-1,len(delta))
        dp_diffusion = vmap(lambda x,y: dpi_diffusion(x,y,theta))(q, p).reshape(-1,len(delta))
                
        return jnp.vstack((dq_diffusion, -dp_diffusion))
        
    return ahs_drift, ahs_diffusion

def ahs_auxillary_model(n, d, k, grad_k, k_tau, grad_k_tau, grad_grad_k_tau, delta, gamma, qT=None):
    
    def z(qi, delta_l, theta):
        
        return jnp.dot(grad_k_tau(qi,delta_l,theta),gamma).reshape(())
    
    def grad_z(qi, delta_l, theta):
        
        return jnp.dot(grad_grad_k_tau(qi,delta_l,theta),gamma).reshape(())
    
    def sigma_l(qi, delta_l, theta):
        
        return gamma*k_tau(qi,delta_l, theta)
    
    #grad_z = grad(z, argnums=0)
    
    def dqi_extra(qi,theta):
        
        z_val = vmap(lambda delta_l: z(qi, delta_l,theta))(delta)
        k_tau_val = vmap(lambda delta_l: k_tau(qi,delta_l, theta))(delta)
                                
        return jnp.dot(z_val,k_tau_val)*gamma
    
    def dpi_extra(qi, theta):
        
        z_val = vmap(lambda delta_l: z(qi, delta_l,theta))(delta)
        grad_k_tau_val = vmap(lambda delta_l: grad_k_tau(qi,delta_l, theta))(delta)
        k_tau_val = vmap(lambda delta_l: k_tau(qi,delta_l, theta))(delta)
        grad_z_val = vmap(lambda delta_l: grad_z(qi, delta_l,theta))(delta)
        
        return (jnp.dot(z_val, grad_k_tau_val)-\
                jnp.dot(k_tau_val, grad_z_val))
            
        #return jnp.zeros(d).reshape(())
                
        
    def dqi_diffusion(qi, theta):
        
        delta_fun = lambda delta_l: sigma_l(qi, delta_l, theta)
        
        return vmap(delta_fun)(delta).T
    
    def dpi_diffusion(qi, theta):
        
        #delta_fun = lambda delta_l: grad_k_tau(qi,delta_l, theta)
        
        #return vmap(delta_fun)(delta).reshape(-1)
        
        return jnp.zeros((len(delta), d)).T
    
    def ahs_betatilde(t, qT, theta):
        
        dq = 1/2*vmap(lambda x: dqi_extra(x,theta))(qT.reshape(n,d)).reshape(-1)
        dp = 1/2*vmap(lambda x: dpi_extra(x,theta))(qT.reshape(n,d)).reshape(-1)
        
        return jnp.hstack((dq, dp))
    
    def ahs_Btilde(t, qT, theta):
        
        def k_qTi(qTi:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
        
            K_val = jnp.zeros((d, n*d))
            K = vmap(lambda x: k(qTi, x,theta))(qT.reshape(n,d)).reshape(-1)
    
            for i in range(d):
                K_val = K_val.at[i,i::d].set(K)
    
            return K_val
        
        K = vmap(lambda x: k_qTi(x,theta))(qT.reshape(n,d)).reshape(-1,n*d)
        
        zero = jnp.zeros_like(K)
        K = jnp.vstack((K, zero))
                        
        return jnp.hstack((jnp.zeros_like(K), K))
    
    def ahs_diffusion_tilde(t, qT, theta):
        
        dq_diffusion = vmap(lambda x: dqi_diffusion(x,theta))(qT.reshape(n,d)).reshape(-1,len(delta))
        dp_diffusion = vmap(lambda x: dpi_diffusion(x,theta))(qT.reshape(n,d)).reshape(-1,len(delta))
                
        return jnp.vstack((dq_diffusion, -dp_diffusion))
       
    if qT is None:
        beta = ahs_betatilde
        B = ahs_Btilde
        sigmatilde = ahs_diffusion_tilde
    else:
        beta = lambda t,theta: ahs_betatilde(t,qT,theta)
        B = lambda t,theta: ahs_Btilde(t,qT, theta)
        sigmatilde = lambda t,theta: ahs_diffusion_tilde(t,qT,theta)
        
    return beta, B, sigmatilde
        
def dH_dq(q, p, grad_k, theta=None)->jnp.ndarray:
    
    if len(q.shape)==1:
        q = q.reshape(-1,1)
        p = p.reshape(-1,1)
        
    def dH_dqi(qi, pi)->jnp.ndarray:
        
        grad_K = vmap(lambda x: grad_k(qi,x,theta))(q)
        inner_prod = jnp.dot(p, pi)
        
        return (grad_K.T * inner_prod).sum(axis=1)
    
    dH = vmap(dH_dqi)(q,p)
    
    return dH

def dH_dp(q, p, k, theta=None)->jnp.ndarray:
    
    if len(q.shape)==1:
        q = q.reshape(-1,1)
        p = p.reshape(-1,1)
        
    def dH_dpi(qi)->jnp.ndarray:
        
        K = vmap(lambda x: k(qi,x,theta))(q)
        
        return (p.T*K).sum(axis=1)
    
    dH = vmap(dH_dpi)(q)
    
    return dH
    
