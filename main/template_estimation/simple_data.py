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

#Link to module folder
import sys
sys.path.insert(1, '../../src/')

#Own modules
import diffusion_bridges as sde_dif
import landmark_models2 as lm
import sim_sp as sp

#%% Functions

#Kernel function
def k(x:jnp.ndarray, y, theta:jnp.ndarray=1.0)->jnp.ndarray:
    
    return jnp.exp(-jnp.dot(x-y,x-y)/(2*(theta**2)))

#Kernel gradient
def grad_k(x:jnp.ndarray, y, theta:jnp.ndarray=1.0)->jnp.ndarray:
    
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
    theta_circ = jnp.exp(sp.sim_normal(mu=theta, sigma=sigma_theta))
    
    return theta_circ

def q_sample_prob(theta_circ, theta):
    
    sigma_theta = 1
    
    return 1/(theta_circ*sigma_theta*jnp.sqrt(2*jnp.pi))*\
        jnp.exp(-(jnp.log(theta_circ)-jnp.log(theta))**2/(2*sigma_theta**2))

def q_prob(theta):
    
    if theta >= 0.01:
        return 0.01*(theta**(-2))
    else:
        return 0
    
    
#Kernel function
def k_tau(x:jnp.ndarray, y, theta:jnp.ndarray=None)->jnp.ndarray:
    
    theta = 0.5
    
    return jnp.exp(-jnp.dot(x-y,x-y)/(2*(theta**2)))

#Kernel gradient
def grad_k_tau(x:jnp.ndarray, y, theta:jnp.ndarray=None)->jnp.ndarray:
    
    theta = 0.5
    
    return -(theta**(-2))*k_tau(x,y,theta)*(x-y)

#Kernel gradient
def grad_grad_k_tau(x:jnp.ndarray, y, theta:jnp.ndarray=None)->jnp.ndarray:
    
    theta = 0.5
    
    return (theta**(-4))*k_tau(x,y,theta)*((x-y)**2)-(theta**(-2))*k_tau(x,y,theta)

#%% Global parameters

I_obs = 5
n = 3
d = 1

q0 = jnp.array([-0.5, 0.0, 0.1])
p0 = jnp.zeros(n)
x0 = jnp.hstack((q0, p0))

#%% Parse arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--save_path', default='simple_models/', 
                        type=str)
    parser.add_argument('--model', default='tv', 
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
    parser.add_argument('--theta', default=1.0, 
                        type=float)
    parser.add_argument('--update_theta', default=0, type=int)
    
    #Iteration parameters
    parser.add_argument('--max_iter', default=10, #20000, 
                        type=int)
    parser.add_argument('--save_step', default=0, 
                        type=int)


    args = parser.parse_args()
    return args

#%% TV-model

def main_tv():
    
    #Arguments
    args = parse_args()
    
    if args.update_theta == 0:
        save_path = args.save_path+'tv'
        theta_update = None
    else:
        save_path = args.save_path+'tv_theta'
        theta_update = None

    gamma = 1/jnp.sqrt(n)*jnp.ones(n)

    time_grid = jnp.arange(args.t0, args.T+args.time_step, args.time_step)
    time_grid = time_grid*(2-time_grid)
    
    SigmaT = args.epsilon**2*jnp.eye(n)
    LT = jnp.hstack((jnp.eye(n), jnp.zeros((n,n))))
    
    b_fun, sigma_fun = lm.tv_model(n, d, k, grad_k, gamma)
    beta_fun, B_fun, sigmatilde_fun = \
        lm.tv_auxillary_model(n, d, k, grad_k, gamma, None)
        
    Wt = sp.sim_Wt(time_grid, n*d, I_obs)
    Xt = vmap(lambda w: sp.sim_sde_euler(x0, lambda t,x: b_fun(t,x,args.theta), 
                                         lambda t,x: sigma_fun(t,x,args.theta), 
                                         w, time_grid))(Wt)
    vT = Xt[:,-1,0:(n*d)]
    
    if args.update_theta == 0:   
        
        beta = beta_fun(0.0,vT[0],args.theta)
        sigmatilde = sigmatilde_fun(0.0, vT[0], args.theta)
        
        beta_funfast = lambda t,vt,theta=args.theta: beta #Since constant in time
        B_funfast = lambda t,vt, theta=args.theta: B_fun(0,vt,theta) #Since constant in time
        sigmatilde_funfast = lambda t,vt,theta=args.theta: sigmatilde #Since constant in time
        b_funsimple = lambda t,x,theta=args.theta : b_fun(t,x,theta)
        sigma_funsimple = lambda t,x,theta=args.theta : sigma_fun(t,x,theta)
    else:
        beta_funfast = lambda t,vt,theta: beta_fun(0.0, vt,theta) #Since constant in time
        B_funfast = lambda t,vt,theta: B_fun(0.0, vt,theta) #Since constant in time
        sigmatilde_funfast = lambda t,vt,theta: sigmatilde_fun(0.0, vt,theta) #Since constant in time
        b_funsimple = lambda t,x,theta : b_fun(t,x,theta)
        sigma_funsimple = lambda t,x,theta : sigma_fun(t,x,theta)
    
    Wt, Xt = sde_dif.landmark_template_qT(vT,
                  SigmaT,
                  LT,
                  b_funsimple,
                  sigma_funsimple,
                  beta_funfast,
                  B_funfast,
                  sigmatilde_funfast,
                  k,
                  pi_prob,
                  time_grid,
                  n = n,
                  d = d,
                  max_iter = args.max_iter,
                  eta=args.eta,
                  deltaq = args.delta,
                  deltap = args.delta,
                  theta = theta_update,
                  q_sample = q_sample,
                  q_sample_prob = q_sample_prob,
                  q_prob = q_prob,
                  backward_method = 'odeint',
                  Wt = None,
                  q0 = None,
                  p0 = p0,
                  update_p0 = False,
                  save_step = args.save_step,
                  save_path = save_path)
    
    return

#%% MS-model

def main_ms():
    
    #Arguments
    args = parse_args()
    
    if args.update_theta == 0:
        save_path = args.save_path+'ms'
        theta_update = None
    else:
        save_path = args.save_path+'ms_theta'
        theta_update = args.theta
    
    gamma = 1/jnp.sqrt(n)*jnp.ones(n)
    lmbda = 1.0

    time_grid = jnp.arange(args.t0, args.T+args.time_step, args.time_step)
    time_grid = time_grid*(2-time_grid)
    
    SigmaT = args.epsilon**2*jnp.eye(n)
    LT = jnp.hstack((jnp.eye(n), jnp.zeros((n,n))))
    
    b_fun, sigma_fun = lm.ms_model(n, d, k, grad_k, lmbda, gamma)
    beta_fun, B_fun, sigmatilde_fun = \
        lm.ms_auxillary_model(n, d, k, grad_k, lmbda, gamma, None)
        
    Wt = sp.sim_Wt(time_grid, n*d, I_obs)
    Xt = vmap(lambda w: sp.sim_sde_euler(x0, lambda t,x: b_fun(t,x,args.theta), 
                                         lambda t,x: sigma_fun(t,x,args.theta), 
                                         w, time_grid))(Wt)
    vT = Xt[:,-1,0:(n*d)]
    
    if args.update_theta == 0:
        
        beta = beta_fun(0.0,vT[0],args.theta)
        sigmatilde = sigmatilde_fun(0.0, vT[0], args.theta)
        
        beta_funfast = lambda t,vt,theta=args.theta: beta #Since constant in time
        B_funfast = lambda t,vt, theta=args.theta: B_fun(0,vt,theta) #Since constant in time
        sigmatilde_funfast = lambda t,vt,theta=args.theta: sigmatilde #Since constant in time
        b_funsimple = lambda t,x,theta=args.theta : b_fun(t,x,theta)
        sigma_funsimple = lambda t,x,theta=args.theta : sigma_fun(t,x,theta)
    else:
        beta_funfast = lambda t,vt,theta: beta_fun(0.0, vt,theta) #Since constant in time
        B_funfast = lambda t,vt,theta: B_fun(0.0, vt,theta) #Since constant in time
        sigmatilde_funfast = lambda t,vt,theta: sigmatilde_fun(0.0, vt,theta) #Since constant in time
        b_funsimple = lambda t,x,theta : b_fun(t,x,theta)
        sigma_funsimple = lambda t,x,theta : sigma_fun(t,x,theta)
    
    _ = sde_dif.landmark_template_qT(vT,
                  SigmaT,
                  LT,
                  b_funsimple,
                  sigma_funsimple,
                  beta_funfast,
                  B_funfast,
                  sigmatilde_funfast,
                  k,
                  pi_prob,
                  time_grid,
                  n = n,
                  d = d,
                  max_iter = args.max_iter,
                  eta=args.eta,
                  deltaq = args.delta,
                  deltap = args.delta,
                  theta = theta_update,
                  q_sample = q_sample,
                  q_sample_prob = q_sample_prob,
                  q_prob = q_prob,
                  backward_method = 'odeint',
                  Wt = None,
                  q0 = None,
                  p0 = p0,
                  update_p0 = False,
                  save_step = args.save_step,
                  save_path = save_path)
    
    return

#%% AHS-model

def main_ahs():
    
    #Arguments
    args = parse_args()
    
    if args.update_theta == 0:
        save_path = args.save_path+'ahs'
        theta_update = None
    else:
        save_path = args.save_path+'ahs_theta'
        theta_update = args.theta

    time_grid = jnp.arange(args.t0, args.T+args.time_step, args.time_step)
    time_grid = time_grid*(2-time_grid)
    
    gamma = jnp.array(0.1*2/jnp.pi)
    delta = jnp.linspace(-2.5, 2.5, 6)
    
    SigmaT = args.epsilon**2*jnp.eye(n)
    LT = jnp.hstack((jnp.eye(n), jnp.zeros((n,n))))
    
    b_fun, sigma_fun = lm.ahs_model(n, d, k, grad_k, k_tau, grad_k_tau, grad_grad_k_tau,
                                    delta, gamma)
    
    beta_fun, B_fun, sigmatilde_fun = lm.ahs_auxillary_model(n, d, k, 
                                                                       grad_k, k_tau, 
                                                                       grad_k_tau, 
                                                                       grad_grad_k_tau,
                                                                       delta, gamma, 
                                                                       None)
    
    Wt = sp.sim_Wt(time_grid, len(delta), I_obs)
    Xt = vmap(lambda w: sp.sim_sde_euler(x0, lambda t,x: b_fun(t,x,args.theta), 
                                         lambda t,x: sigma_fun(t,x,args.theta), 
                                         w, time_grid))(Wt)
    vT = Xt[:,-1,0:(n*d)]
    
    if args.update_theta == 0:
        beta_funfast = lambda t,vt,theta=args.theta: beta_fun(0.0,vt,theta) #Since constant in time
        B_funfast = lambda t,vt, theta=args.theta: B_fun(0,vt,theta) #Since constant in time
        sigmatilde_funfast = lambda t,vt,theta=args.theta: sigmatilde_fun(0.0,vt,theta) #Since constant in time
        b_funsimple = lambda t,x,theta=args.theta : b_fun(t,x,theta)
        sigma_funsimple = lambda t,x,theta=args.theta : sigma_fun(t,x,theta)
    else:
        beta_funfast = lambda t,vt,theta: beta_fun(0.0, vt,theta) #Since constant in time
        B_funfast = lambda t,vt,theta: B_fun(0.0, vt,theta) #Since constant in time
        sigmatilde_funfast = lambda t,vt,theta: sigmatilde_fun(0.0, vt, theta) #Since constant in time
        b_funsimple = lambda t,x,theta : b_fun(t,x,theta)
        sigma_funsimple = lambda t,x,theta : sigma_fun(t,x,theta)
    
    _ = sde_dif.landmark_template_qT(vT,
                  SigmaT,
                  LT,
                  b_funsimple,
                  sigma_funsimple,
                  beta_funfast,
                  B_funfast,
                  sigmatilde_funfast,
                  k,
                  pi_prob,
                  time_grid,
                  n = n,
                  d = d,
                  max_iter = args.max_iter,
                  eta=args.eta,
                  deltaq = args.delta,
                  deltap = args.delta,
                  theta = theta_update,
                  q_sample = q_sample,
                  q_sample_prob = q_sample_prob,
                  q_prob = q_prob,
                  backward_method = 'odeint',
                  Wt = None,
                  q0 = None,
                  p0 = p0,
                  update_p0 = False,
                  save_step = args.save_step,
                  save_path = save_path)
    
    return


#%% Calling main

if __name__ == '__main__':
    args = parse_args()
    
    if args.model == 'ahs':
        main_ahs()
    elif args.model == 'tv':
        main_tv()
    else:
        main_ms()







