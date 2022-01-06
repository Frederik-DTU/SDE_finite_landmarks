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

#Parse arguments
import argparse

#Own modules
from finite_landmark_models import landmark_models
from sde_approx import sde_finite_landmarks
from plot_landmarks import plot_landmarks


#%% Functions

#Kernel function
def k(x:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
    
    theta = 1.0
    
    return jnp.exp(-jnp.dot(x,x)/((2*theta)**2))

#Kernel gradient
def grad_k(x:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
    
    theta = 1.0
    
    return (theta**(-2))*k(x,theta)*x

#%% Parse arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--save_path', default='Model_output/1d_landmarks', 
                        type=str)
    
    #Hyper-parameters
    parser.add_argument('--lambda_', default=1.0, 
                        type=float)
    parser.add_argument('--time_step', default=0.01, #0.001,
                        type=float)
    parser.add_argument('--t0', default=0.0, 
                        type=float)
    parser.add_argument('--T', default=1.0, 
                        type=float)
    parser.add_argument('--n_sim', default=10, 
                        type=int)
    
    #Iteration parameters
    parser.add_argument('--seed', default=2712, 
                        type=int)

    args = parser.parse_args()
    return args

#%% main

def main():
    
    #Arguments
    args = parse_args()
    
    sde = sde_finite_landmarks(seed=args.seed)
    land = landmark_models(k, grad_k)
    plt_land = plot_landmarks()
    n = 3
    d = 1

    time_grid = jnp.arange(args.t0, args.T+args.time_step, args.time_step)
    time_grid = time_grid*(2-time_grid)
    
    q0 = jnp.array([-0.5, 0.0, 0.1]).reshape(n,d)
    p0 = jnp.array([-1.0, -0.5, 0.5]).reshape(n,d)
    qT = jnp.array([-0.5, 0.2, 1.0]).reshape(n,d)
    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
    

    gamma = 1/jnp.sqrt(n)*jnp.ones(n)
    
    b_fun, sigma_fun = land.get_ms_model(gamma, dim = [2*n, d], lambda_=args.lambda_)
    betatilde_fun, Btilde_fun, sigmatilde_fun = land.get_ms_approx_model(gamma, 
                                                                     [2*n,d], qT, 
                                                                     args.lambda_)   
    
    betatilde = betatilde_fun(0) #Since constant in time
    Btilde = Btilde_fun(0) #Since constant in time
    sigmatilde = sigmatilde_fun(0) #Since constant in time
    
    btilde_fun = lambda t,x,theta : betatilde+jnp.dot(Btilde,x)
    sigmatilde_fun = lambda t,x,theta : sigmatilde
    
    t, qt, pt = land.landmark_shooting_ivp_rk45(x0.reshape(-1,1), b_fun, time_grid, None)
    plt_land.plot_1d_landmarks_ivp(t, qt, pt, title='Landmark Shooting MS')
    
    t, qt, pt = land.landmark_matching_bfgs(q0, qt[-1].reshape(n,d), b_fun, time_grid, None, p0)
    #t, qt, pt = land.landmark_matching_bfgs(q0, qT, b_fun, time_grid, None, p0)
    plt_land.plot_1d_landmarks_bvp(t, qt, pt, title='Exact Matching MS')
    
    t, sim = sde.sim_sde(x0, b_fun, sigma_fun, n_sim=args.n_sim, grid=time_grid)
    
    qt_sim = sim[:,:,:n]
    pt_sim = sim[:,:,n:]
    plt_land.plot_1d_landmarks_realisations(t, qt_sim, pt_sim, title='MS-Model')
    
    sde.reset_seed(args.seed)
    
    t, sim = sde.sim_sde(x0, btilde_fun, sigmatilde_fun, n_sim=args.n_sim, grid=time_grid)
    
    qt_sim = sim[:,:,:n]
    pt_sim = sim[:,:,n:]
    plt_land.plot_1d_landmarks_realisations(t, qt_sim, pt_sim, title='Auxillary MS-Model')

    
    return


#%% Calling main

if __name__ == '__main__':
    main()
