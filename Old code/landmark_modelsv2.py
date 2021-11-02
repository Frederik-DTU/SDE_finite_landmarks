#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 13:49:17 2021

@author: Frederik
"""

#%% Sources

"""
solve_bvp:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html#r25f8479e577a-3
solve_ivp:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
gradient in jax:
https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
"""

#%% Modules

#Standard modules
import numpy as np
import jax.numpy as jnp
from jax import grad
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve

#Typing
from typing import Callable, List

#%% Landmark class to solve Hamiltonian equations


class landmark_models(object):
    def __init__(self, 
                 k:Callable[[jnp.ndarray], jnp.ndarray] = None
                 ):
        
        self.k = k #The kernel is a identitfy matrix, i.e. K(y)=k(y)*Id(d)
        self.grad_k = grad(self.hamiltonian_formulation)
        self.bc_q0 = None
        self.bc_qT = None
        self.dim = None
        self.q0 = None
        self.qT = None
        
    def hamiltonian_formulation(self, y:jnp.ndarray)->jnp.ndarray:
        
        n = int(y.shape[0]/2)
        
        q = y[0:n]
        p = y[n:]
        
        sum_k = 0.0
        for i in range(n):
            for j in range(n):
                sum_k += jnp.dot(p[i], p[j])*self.k(q[i]-q[j])
                
        return 1/2*sum_k
    
    def hamiltonian_grad(self, t:jnp.ndarray, y:jnp.ndarray)->jnp.ndarray:
        
        if len(y.shape)>1:
            grad = jnp.array([self.grad_k(yi) for yi in y])
        else:
            grad = self.grad_k(y)
                
        grad_new = grad
        
        grad_new = grad_new.at[0:self.dim].set(grad[self.dim:])
        grad_new = grad_new.at[self.dim:].set(-grad[0:self.dim])
        
        return grad_new
    
    def ivp_hamiltonian(self, y0:jnp.ndarray, n_grid:int=10, T:float=1)->jnp.ndarray:
        
        x_mesh = jnp.linspace(0, T, n_grid)
        self.dim = int(y0.shape[0]/2)
        
        sol = solve_ivp(self.hamiltonian_grad, [0,T], y0, t_eval = x_mesh)
        
        qt = sol.y[0:self.dim]
        pt = sol.y[self.dim:]
        
        if sol.status != 0:
            print("WARNING: sol.status is %d" % sol.status)
        
        return sol.t, qt, pt
    
    def __hamiltonian_bc(self, y0:jnp.ndarray, yT:jnp.ndarray)->List:
        
        bc = []
                
        for i in range(self.dim):
            bc.append(y0[i]-self.bc_q0[i])
            bc.append(yT[i]-self.bc_qT[i])
                        
        return bc
    
    def bvp_hamiltonian(self, q0:jnp.ndarray, qT:jnp.ndarray, 
                        y0_grid:jnp.ndarray,
                        n_grid:int=10, T:float=1)->jnp.ndarray:
        
        self.bc_q0 = q0
        self.bc_qT = qT
        self.dim = int(q0.shape[0])
        
        x_mesh = jnp.linspace(0, T, n_grid)
        
        sol = solve_bvp(self.hamiltonian_grad,
                        self.__hamiltonian_bc,
                        x_mesh, y0_grid)
        
        qt = sol.y[0:self.dim]
        pt = sol.y[self.dim:]
        
        if sol.status != 0:
            print("WARNING: sol.status is %d" % sol.status)
            print("-", sol.message)
                
        return sol.x, qt, pt
    
    def bvp_hamiltonian2(self, q0:jnp.ndarray, qT:jnp.ndarray, 
                        p0_init:jnp.ndarray,
                        n_grid:int=10, T:float=1)->jnp.ndarray:
    
        
        self.q0 = q0
        self.qT = qT
        self.dim = q0.shape[0]
                
        p0 = fsolve(self.func, p0_init)
        
        y0 = jnp.concatenate((self.q0, p0))
        sol = self.ivp_hamiltonian(y0)
        
        qt = sol.y[0:self.dim]
        pt = sol.y[self.dim:]
                
        return sol.t, qt, pt
    
    def func(self, p0:jnp.ndarray):
        
        
        y0 = jnp.concatenate((self.q0, p0))
        _, qt, _ = self.ivp_hamiltonian(y0)
        
        print(qt[:,-1]-self.qT)
        return qt[:,-1]-self.qT
    
    def mh_drift(self, t:jnp.ndarray, y:jnp.ndarray, lam:float=5)->jnp.ndarray:
        
        if len(y.shape)>1:
            grad = jnp.array([self.grad_k(yi) for yi in y])
        else:
            grad = self.grad_k(y)
        
        n = int(y.shape[0]/2)
        grad = grad.at[n:].set(grad[n:]-lam*grad[0:n])
        
        return grad
    
    def ahs_noise(self, t:jnp.ndarray, y:jnp.ndarray, lam:float=5)->jnp.ndarray:
        
        return
        
        
        
    
#%% Testing


import matplotlib.pyplot as plt

def k(x):
    return jnp.exp(-jnp.dot(x,x)/2)


test_grad = grad(k)
ham = landmark_models(k)

N = 5
q0 = jnp.linspace(-.5,.5,N)
p0 = jnp.linspace(-0.3,-0.1,N)

q0 = jnp.array([-0.5, 0.0, 0.1])
p0 = jnp.linspace(-1,1,3)
y0 = jnp.concatenate((q0, p0))

ham_form = ham.hamiltonian_formulation(y0)

t = jnp.array([1.0, 2.0])
ham_grad = ham.hamiltonian_grad(t, y0)

t_ivp, qt_ivp, pt_ivp = ham.ivp_hamiltonian(y0, n_grid=100)

plt.figure()
plt.plot(t_ivp, qt_ivp.transpose())
plt.title("q")
plt.ylabel("Position")
plt.xlabel("t")
plt.legend(["q1(t)", "q2(t)", "q3(t)"])
plt.grid()
plt.figure()
plt.plot(t_ivp, pt_ivp.transpose())
plt.title("p")
plt.ylabel("Moment")
plt.xlabel("t")
plt.legend(["p1(t)", "p2(t)", "p3(t)"])
plt.grid()


#q0 = q0
#qT = qt_ivp[:,-1]
#y0_grid = np.concatenate((qt_ivp, pt_ivp))
#y0_grid = jnp.ones((2*3, 10))
#t_bvp, qt_bvp, pt_bvp = ham.bvp_hamiltonian(q0, qT, y0_grid)

#p0_init = pt_ivp[:,0]
#p0_init = [-0.95, 0.95]

#q0 = jnp.array([-0.5, 0.0, 0.1])
#qT = jnp.array([-0.5, 0.2, 1.0])
#p0_init = jnp.array([-1.0, 1.0, 0.5])
#t_bvp, qt_bvp, pt_bvp = ham.bvp_hamiltonian2(q0, qT, p0_init)
#y0_grid = jnp.zeros((2*3, 10))
#t_bvp, qt_bvp, pt_bvp = ham.bvp_hamiltonian(q0, qT, y0_grid)

#plt.figure()
#plt.plot(t_bvp, qt_bvp.transpose())
    
#x = np.linspace(-1,1,100)
#gx = test_grad(x)
#plt.figure()
#plt.plot(gx)



    