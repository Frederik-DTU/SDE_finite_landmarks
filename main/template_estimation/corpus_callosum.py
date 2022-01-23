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

import sys

from scipy import io

#Own modules
import diffusion_bridges as sde_dif
import landmark_models2 as lm
import sim_sp as sp

#%% Functions

#Kernel function
def k(x:jnp.ndarray, y, theta:jnp.ndarray=None)->jnp.ndarray:
    
    theta = 1.0
    
    return jnp.exp(-jnp.dot(x-y,x-y)/(2*(theta**2)))

#Kernel gradient
def grad_k(x:jnp.ndarray, y, theta:jnp.ndarray=None)->jnp.ndarray:
    
    theta = 1.0
    
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
    
    q0 = q0.reshape(n,d)
    p0 = p0.reshape(n,d)
    
    Sigma = kmom*jnp.linalg.inv(vmap(compute_row)(q0))
    mu = jnp.zeros(n)
    
    pi = jnp.prod(multivariate_normal.pdf(p0.T, mean=mu, cov = Sigma))
    
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

sys.path.insert(1, '../src')

data=io.loadmat('Data/dataM-corpora-callosa-ipmi-full.mat')
vi = data['vi'] # corpora callosa
Nobs = vi.shape[1]
N0 = int(vi.shape[0]/2-1)
qs = vi[0:-2,:].T.reshape((Nobs,N0,2))

    
n = N0
d = 2
I_obs = 5

p0 = jnp.zeros((n,d))
qT = qs[0:I_obs]
vT = qT

#%% Parse arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--save_path', default='corpus_callosum_models/', 
                        type=str)
    parser.add_argument('--model', default='ahs', 
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
    parser.add_argument('--theta', default=0.5, 
                        type=float)
    
    #Iteration parameters
    parser.add_argument('--max_iter', default=2, #20000, 
                        type=int)
    parser.add_argument('--save_step', default=2, 
                        type=int)


    args = parser.parse_args()
    return args

#%% TV-model

def main_tv():
    
    #Arguments
    args = parse_args()
    
    if args.theta is None:
        save_path = args.save_path+'tv'
    else:
        save_path = args.save_path+'tv_theta'

    gamma = 1/jnp.sqrt(n)*jnp.ones(n*d)

    time_grid = jnp.arange(args.t0, args.T+args.time_step, args.time_step)
    time_grid = time_grid*(2-time_grid)
    
    SigmaT = args.epsilon**2*jnp.eye(n*d)
    LT = jnp.hstack((jnp.eye(n*d), jnp.zeros((n*d,n*d))))
    
    b_fun, sigma_fun = lm.tv_model(n, d, k, grad_k, gamma)
    beta_fun, B_fun, sigmatilde_fun = \
        lm.tv_auxillary_model(n, d, k, grad_k, gamma, None)
    
    if args.theta is None:   
        
        beta = beta_fun(0.0,vT[0],None)
        sigmatilde = sigmatilde_fun(0.0, vT[0], None)
        
        beta_funfast = lambda t,vt,theta=None: beta #Since constant in time
        B_funfast = lambda t,vt, theta=None: B_fun(0,vt,None) #Since constant in time
        sigmatilde_funfast = lambda t,vt,theta=None: sigmatilde #Since constant in time
        b_funsimple = lambda t,x,theta=None : b_fun(t,x,theta)
        sigma_funsimple = lambda t,x,theta=None : sigma_fun(t,x,theta)
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
                  theta = args.theta,
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
    
    if args.theta is None:
        save_path = args.save_path+'ms'
    else:
        save_path = args.save_path+'ms_theta'
    
    gamma = 1/jnp.sqrt(n)*jnp.ones(n*d)
    lmbda = 1.0

    time_grid = jnp.arange(args.t0, args.T+args.time_step, args.time_step)
    time_grid = time_grid*(2-time_grid)
    
    SigmaT = args.epsilon**2*jnp.eye(n*d)
    LT = jnp.hstack((jnp.eye(n*d), jnp.zeros((n*d,n*d))))
    
    b_fun, sigma_fun = lm.ms_model(n, d, k, grad_k, lmbda, gamma)
    beta_fun, B_fun, sigmatilde_fun = \
        lm.ms_auxillary_model(n, d, k, grad_k, lmbda, gamma, None)
    
    if args.theta is None:   
        
        beta = beta_fun(0.0,vT[0],None)
        sigmatilde = sigmatilde_fun(0.0, vT[0], None)
        
        beta_funfast = lambda t,vt,theta=None: beta #Since constant in time
        B_funfast = lambda t,vt, theta=None: B_fun(0,vt,None) #Since constant in time
        sigmatilde_funfast = lambda t,vt,theta=None: sigmatilde #Since constant in time
        b_funsimple = lambda t,x,theta=None : b_fun(t,x,theta)
        sigma_funsimple = lambda t,x,theta=None : sigma_fun(t,x,theta)
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
                  theta = args.theta,
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
    
    if args.theta is None:
        save_path = args.save_path+'ahs'
    else:
        save_path = args.save_path+'ahs_theta'

    time_grid = jnp.arange(args.t0, args.T+args.time_step, args.time_step)
    time_grid = time_grid*(2-time_grid)
    
    gamma = jnp.array([0.1*2/jnp.pi, 0.1*2/jnp.pi])
    delta = jnp.vstack((jnp.linspace(-2.5, 2.5, 6),jnp.linspace(-2.5, 2.5, 6))).T
    
    SigmaT = args.epsilon**2*jnp.eye(n*d)
    LT = jnp.hstack((jnp.eye(n*d), jnp.zeros((n*d,n*d))))
    
    b_fun, sigma_fun = lm.ahs_model(n, d, k, grad_k, k_tau, grad_k_tau, grad_grad_k_tau,
                                    delta, gamma)
    
    beta_fun, B_fun, sigmatilde_fun = lm.ahs_auxillary_model(n, d, k, 
                                                                       grad_k, k_tau, 
                                                                       grad_k_tau, 
                                                                       grad_grad_k_tau,
                                                                       delta, gamma, 
                                                                       None)
    
    if args.theta is None:        
        beta_funfast = lambda t,vt,theta=None: beta_fun(0.0,vt,None) #Since constant in time
        B_funfast = lambda t,vt, theta=None: B_fun(0,vt,None) #Since constant in time
        sigmatilde_funfast = lambda t,vt,theta=None: sigmatilde_fun(0.0,vt,None) #Since constant in time
        b_funsimple = lambda t,x,theta=None : b_fun(t,x,theta)
        sigma_funsimple = lambda t,x,theta=None : sigma_fun(t,x,theta)
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
                  theta = args.theta,
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