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

#From scipy
from scipy.stats import multivariate_normal

#Parse arguments
import argparse

#Own modules
from finite_landmark_models import landmark_models
from sde_approx import sde_finite_landmarks


#%% Functions

#Kernel function
def k(x:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
    
    theta = 1.0
    
    return jnp.exp(-jnp.dot(x,x)/((2*theta)**2))

#Kernel gradient
def grad_k(x:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
    
    theta = 1.0
    
    return (theta**(-2))*k(x,theta)*x

#Kernel for prior
def pi_kernel(x:jnp.ndarray)->jnp.ndarray:
    
    theta = 1.0
    
    return jnp.exp(-jnp.dot(x,x)/((2*theta)**2))

#Prior on landmarks
def pi_x0(q0:jnp.ndarray, p0:jnp.ndarray)->jnp.ndarray:
    
    kmom = 100
    
    Sigma = kmom/jnp.apply_along_axis(pi_kernel, -1, q0)
    mu = jnp.zeros_like(p0.reshape(-1))
        
    pi = multivariate_normal.pdf(p0.reshape(-1), mean=mu,
                                          cov = Sigma)
    
    return pi

#%% Parse arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--save_path', default='Model_output/1d_landmarks', 
                        type=str)
    
    #Hyper-parameters
    parser.add_argument('--eta', default=0.1,  #0.1
                        type=float)
    parser.add_argument('--delta', default=0.1, #0.1 
                        type=float)
    parser.add_argument('--lambda_', default=1.0, 
                        type=float)
    parser.add_argument('--epsilon', default=0.001,
                        type=float)
    parser.add_argument('--time_step', default=0.01, #0.001,
                        type=float)
    parser.add_argument('--t0', default=0.0, 
                        type=float)
    parser.add_argument('--T', default=1.0, 
                        type=float)
    
    #Iteration parameters
    parser.add_argument('--seed', default=2712, 
                        type=int)
    parser.add_argument('--max_iter', default=20000, #20000, 
                        type=int)
    parser.add_argument('--save_hours', default=1.0, 
                        type=float)

    #Continue training or not
    parser.add_argument('--load_model_path', default='Model_output/1d_landmarks_ite_1000.npy',
                        type=str)


    args = parser.parse_args()
    return args

#%% main

def main():
    
    #Arguments
    args = parse_args()
    
    sde = sde_finite_landmarks(seed=args.seed)
    land = landmark_models(k, grad_k)
    n = 3
    d = 1

    time_grid = jnp.arange(args.t0, args.T+args.time_step, args.time_step)
    time_grid = time_grid*(2-time_grid)
    
    q0 = jnp.array([-0.5, 0.0, 0.1]).reshape(n,d)
    p0 = jnp.array([-0.1, 0.0, 0.1]).reshape(n,d)
    qT = jnp.array([-0.5, 0.2, 1.0]).reshape(n,d)
    vT = qT
    
    #SigmaT = (1/args.epsilon)**2*jnp.eye(n)
    SigmaT = args.epsilon**2*jnp.eye(n)
    gamma = 1/jnp.sqrt(n)*jnp.ones(n)
    LT = jnp.hstack((jnp.eye(n), jnp.zeros((n,n))))
    
    b_fun, sigma_fun = land.get_ms_model(gamma, dim = [2*n, d], lambda_=args.lambda_)
    betatilde_fun, Btilde_fun, sigmatilde_fun = land.get_ms_approx_model(gamma, 
                                                                     [2*n,d], qT, 
                                                                     args.lambda_)   
    
    betatilde = betatilde_fun(0) #Since constant in time
    Btilde = Btilde_fun(0) #Since constant in time
    sigmatilde = sigmatilde_fun(0) #Since constant in time
    
    betatilde_funfast = lambda t,theta: betatilde #Since constant in time
    Btilde_funfast = lambda t,theta: Btilde #Since constant in time
    sigmatilde_funfast = lambda t,theta: sigmatilde #Since constant in time
        
    _, Xt = sde.approx_p0(q0, p0, vT, SigmaT, LT, 
                        b_fun, sigma_fun, 
                        betatilde_funfast, Btilde_funfast, sigmatilde_funfast, 
                        pi_x0,
                        time_grid = time_grid,
                        max_iter = args.max_iter,
                        eta = args.eta,
                        delta = args.delta,
                        save_path=args.save_path,
                        save_hours=args.save_hours)
    
    save_file = args.save_path+'_'+str(args.max_iter)
    jnp.save(save_file, Xt)
    
    return


#%% Calling main

if __name__ == '__main__':
    main()
