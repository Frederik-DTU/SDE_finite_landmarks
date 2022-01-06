#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 19:54:28 2021

@author: Frederik
"""

#%% Sources
    
    
#%% Modules

#Standard modules
import numpy as np
import jax.numpy as jnp
from jax import grad
from scipy.integrate import solve_bvp, solve_ivp

#Typing
from typing import Callable, Tuple, List

#%% Landmark class

class landmark_models(object):
    def __init__(self,
                 k:Callable[[jnp.ndarray], jnp.ndarray] = None):
        
        self.k = k #The kernel is a identitfy matrix, i.e. K(y)=k(y)*Id(d)
        self.grad_k = grad(k) #The gradient of the scalar kernel
        self.grad_k2 = grad(self.hamiltonian_formulation)
        self.bc_q0 = None
        self.bc_qT = None
        self.dim = None
        
    def compute_gradient(self, q:jnp.ndarray)->jnp.ndarray:
    
        if q.reshape(-1).shape[0]==1:
            return self.grad_k(q)
        else:
            return jnp.array([self.grad_k(xi) for xi in q])
        
        
    def hamiltonian_formulation(self, q:jnp.ndarray, p:jnp.ndarray)->jnp.ndarray:
        
        n = q.shape[0]
        sum_k = 0.0
        for i in range(n):
            for j in range(n):
                sum_k += jnp.dot(p[i], p[j])*self.k(q[i]-q[j])
                
        return 1/2*sum_k
    
    def hamiltoan_equations_dqi(self, qi:jnp.ndarray, q:jnp.ndarray, 
                                p:jnp.ndarray)->jnp.ndarray:
        
        n = q.shape[0]
        dqi = jnp.zeros_like(qi)
        for j in range(n):
            dqi += p[j]*self.k(qi-q[j])

        return dqi
    
    def hamiltoan_equations_dpi(self, pi:jnp.ndarray, qi:jnp.ndarray, 
                                q:jnp.ndarray, p:jnp.ndarray)->jnp.ndarray:
        
        n = q.shape[0]
        dpi = jnp.zeros_like(pi)
        for j in range(n):
            dpi += jnp.dot(pi, p[j])*self.compute_gradient(qi-q[j])
        
        return -dpi
    
    def __hamiltonian_fun(self, t:jnp.ndarray, y:jnp.ndarray)->jnp.ndarray:
        
        q = y[0:self.dim]
        p = y[self.dim:]
        dq = jnp.zeros_like(q)
        dp = jnp.zeros_like(p)
        
        for i in range(self.dim):
            dq = dq.at[i].set(self.hamiltoan_equations_dqi(q[i], q, p))
            
        for i in range(self.dim):
            dp = dp.at[i].set(self.hamiltoan_equations_dpi(p[i], q[i], q, p))
        
        return jnp.concatenate((dq, dp))
    
    def __hamiltonian_bc(self, y0:jnp.ndarray, yT:jnp.ndarray)->List:
        
        bc = []
                
        for i in range(self.dim):
            bc.append(y0[i]-self.bc_q0[i])
            bc.append(yT[i]-self.bc_qT[i])
                        
        return bc
    
    def ivp_hamiltonian(self, y0:jnp.ndarray, n_grid:int=10, T:int=1)->jnp.ndarray:
        
        x_mesh = jnp.linspace(0, T, n_grid)
        self.dim = int(y0.shape[0]/2)
        
        sol = solve_ivp(self.__hamiltonian_fun, [0,T], y0, t_eval = x_mesh)
        
        qt = sol.y[0:self.dim]
        pt = sol.y[self.dim:]
        
        return sol
    
    def bvp_hamiltonian(self, q0:jnp.ndarray, qT:jnp.ndarray, 
                        y0_grid:jnp.ndarray,
                        n_grid:int=10, T:int=1)->jnp.ndarray:
        
        self.bc_q0 = q0
        self.bc_qT = qT
        n = int(q0.shape[0])
        self.dim = n
        
        x_mesh = jnp.linspace(0, T, n_grid)
        
        sol = solve_bvp(self.__hamiltonian_fun,
                        self.__hamiltonian_bc,
                        x_mesh, y0_grid)
        
        qt = sol.y[0:n]
        pt = sol.y[n:]
                
        return sol
    
    def sim_tv_model(self, T, n):
        
        return 2
    
    def sim_bm(self, T, n):
        
        dt = T/n
        Wt = np.zeros(n)
        sam_norm = np.random.normal(0, np.sqrt(dt), n)
        
        for j in range(1,n):
            Wt[j] += Wt[j-1]+sam_norm[j-1]
            
        return Wt
    
    def test(self, pi:jnp.ndarray, qi:jnp.ndarray, 
                                q:jnp.ndarray, p:jnp.ndarray):
        
        self.grad_k2
        
        return 
    
#%% Test
#Gaussian kernel
#def k(x):
    
#    if len(x.shape)>1:
#        return jnp.exp(-jnp.linalg.norm(x, axis=1)**2/(2))
#    else:
#        return jnp.exp(-jnp.linalg.norm(x)**2/(2))
    
def k(x):
    return jnp.exp(-x*x/2)


test_jax = grad(k)
N = 3

ham = landmark_models(k)
q0 = jnp.array([10,0.2,0.3])
p0 = jnp.array([0.5, 0.75, 1.0])
y0 = jnp.concatenate((q0, p0))


q0 = jnp.linspace(-.5,.5,N)
p0 = jnp.linspace(-1,1,N)
y0 = jnp.concatenate((q0, p0))




test = ham.hamiltonian_formulation(q0, p0)
test = ham.hamiltoan_equations_dqi(q0[0], q0, p0)
test = ham.hamiltoan_equations_dpi(q0[0], p0[0], q0, p0)
#sol = ham.ivp_hamiltonian(y0)


y0_init = jnp.zeros((2*N,10))
q0 = q0
qT = jnp.array([-1.00136118,  0.        ,  1.00136118])

#sol = ham.bvp_hamiltonian(q0, qT, y0_init)
            
        
#import matplotlib.pyplot as plt

#plt.plot(sol.y[3])



def test(x):
    
    n = int(x.shape[0]/2)
    y = x[n:]
    x1 = x[0:n]
    
    return jnp.dot(x1,y)


grad_test = grad(test)

test1 = jnp.array([1.0,2.0, 3.0, 4.0])
grad_test(test1)
        
#%% testing
"""
import jax.numpy as jnp
from jax import grad
from jax import random

q1 = jnp.array([1.0, 2.0, 3.0])
q2 = jnp.array([4.0, 5.0, 6.0])

#Gaussian kernel
def k(x):
    
    if len(x.shape)>1:
        return jnp.exp(-jnp.linalg.norm(x, axis=1)**2/(2))
    else:
        return jnp.exp(-jnp.linalg.norm(x)**2/(2))
    
x = np.array([[1.,2,3],[4.,5,6]])
test = k(x)

q0 = jnp.array([[1.0, 2., 3],
               [2., 4., 5.],
               [6., 7., 8.]])


k(q1-q2)

test_jax = grad(k)
print(test_jax(q1-q2))

import torch, torchvision

def k(x):
    
    if len(x.shape)>1:
        return torch.exp(-torch.norm(x, dim=1)**2/(2))
    else:
        return torch.exp(-torch.norm(x)**2/(2))
    
q1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
q2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
q0 = torch.tensor([[1.0, 2., 3],
               [2., 4., 5.],
               [6., 7., 8.]], requires_grad=True)

test = k(q1-q2)
test.backward()
print(q2.grad)

test = k(q0)
test[2].backward()
print(q0.grad)

M = 5
N = 2
q = jnp.vstack((np.linspace(-.5,.5,10),np.zeros(10))).T
"""