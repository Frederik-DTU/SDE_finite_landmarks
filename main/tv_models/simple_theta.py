#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 21:08:20 2021

@author: root
"""

#%% Sources:
    
    

#%% Modules

#JAX
import jax.numpy as jnp
from jax import vmap

#From scipy
from scipy.stats import multivariate_normal

#Parse arguments
import argparse

#Own modules
import diffusion_bridges as sde_dif
import landmark_models2 as lm
import sim_sp as sp

#%% Functions

#Kernel function
def k(x:jnp.ndarray, y, theta:jnp.ndarray=None)->jnp.ndarray:
    
    theta = float(theta)

    return jnp.exp(-jnp.dot(x-y,x-y)/(2*(theta**2)))

#Kernel gradient
def grad_k(x:jnp.ndarray, y, theta:jnp.ndarray=None)->jnp.ndarray:
    
    theta = float(theta)
    
    return -(theta**(-2))*k(x,y,theta)*(x-y)

#Kernel for prior
def pi_kernel(x:jnp.ndarray)->jnp.ndarray:
    
    theta = 1.0
    
    return jnp.exp(-jnp.dot(x,x)/(2*(theta**2)))

#Prior on landmarks
def pi_prob(q0:jnp.ndarray, p0:jnp.ndarray)->jnp.ndarray:
    
    def compute_row(qi):
        
        return vmap(pi_kernel)(q0-qi)
    
    kmom = 100
    
    Sigma = kmom*jnp.linalg.inv(vmap(compute_row)(q0))
    mu = jnp.zeros_like(p0.reshape(-1))
    
    pi = multivariate_normal.pdf(p0.reshape(-1), mean=mu,
                                          cov = Sigma)
    
    return pi

def q_sample(theta):
    
    sigma_theta = 1
    theta_circ = jnp.exp(sp.sim_normal(mu=jnp.log(theta), sigma=sigma_theta))
    
    return theta_circ

def q_prob(theta):
    
    if theta >= 0.1:
        return 0.1*(theta**(-2))
    else:
        return 0
    

#%% Parse arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--save_path', default='1d_saved/', 
                        type=str)
    
    #Hyper-parameters
    parser.add_argument('--eta', default=0.98, #Should close to 1
                        type=float)
    parser.add_argument('--delta', default=0.001, #Should be low
                        type=float)
    parser.add_argument('--epsilon', default=0.001,#0.001,
                        type=float)
    parser.add_argument('--time_step', default=0.01, #0.001,
                        type=float)
    parser.add_argument('--t0', default=0.0, 
                        type=float)
    parser.add_argument('--T', default=1.0, 
                        type=float)
    
    #Iteration parameters
    parser.add_argument('--max_iter', default=100, #20000, 
                        type=int)
    parser.add_argument('--save_step', default=0, 
                        type=int)


    args = parser.parse_args()
    return args

#%% main

def main():
    
    #Arguments
    args = parse_args()
    
    n = 3
    d = 1

    time_grid = jnp.arange(args.t0, args.T+args.time_step, args.time_step)
    time_grid = time_grid*(2-time_grid)
    
    q0 = jnp.array([-0.5, 0.0, 0.1])
    qT = jnp.array([-0.5, 0.2, 1.0])
    vT = qT.reshape(-1)
    
    SigmaT = args.epsilon**2*jnp.eye(n)
    gamma = 1/jnp.sqrt(n)*jnp.ones(n)
    LT = jnp.hstack((jnp.eye(n), jnp.zeros((n,n))))
    
    b_fun, sigma_fun = lm.tv_model(n, d, k, grad_k, gamma)
    betatilde_fun, Btilde_fun, sigmatilde_fun = \
        lm.tv_auxillary_model(n, d, k, grad_k, gamma, qT)
    
    theta, Wt, Xt = sde_dif.landmark_segment2(q0, vT, SigmaT, LT, b_fun, sigma_fun, 
                                  betatilde_fun, 
                                  Btilde_fun, sigmatilde_fun, pi_prob, 
                                  max_iter = args.max_iter,
                                  time_grid = time_grid,
                                  eta=args.eta,
                                  delta=args.delta,
                                  theta = 1.0,
                                  q_sample = q_sample,
                                  q_prob = q_prob,
                                  backward_method = 'odeint',
                                  save_step = args.save_step,
                                  save_path = args.save_path)
    
    jnp.savez(args.save_path+'Wt_Xt_'+str(args.max_iter), Wt=Wt, Xt=Xt)
    
    print(Xt[-1])
    print(theta[-1])
    
    return


#%% Calling main

if __name__ == '__main__':
    main()







