#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 17:26:30 2021

@author: Frederik
"""

#%% Sources

#%% Modules used

#JAX Modules
import jax.numpy as jnp
from jax import grad

#From scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


from sde_approx import sde_finite_landmarks

#Typing declaration of variables
from typing import Callable, List, Tuple

#%% Class

class landmark_models(object):
    """
    This class estimmates SDE's on finite dimensional landmark spaces,
    where the end point is partially observed using guided proposals.

    ...

    Attributes
    ----------
    seed : int
        the seed value for sampling of random numbers

    Methods
    -------
    reset_seed(seed:int)->type(None)
        Updates the seed value
        
    sim_Wt(n_sim:int=10, n_steps:int=100, dim:int=1,t0:float = 0.0, 
           T:float = 1.0)->Tuple[jnp.ndarray, jnp.ndarray]
        Simulates Wiener process
        
    sim_multi_normal(mu:jnp.ndarray = jnp.zeros(2),sigma:jnp.ndarray=jnp.eye(2),
               dim:List[int] = [1])
        Simulates multivariate normal distribution
        
    multi_normal_pdf(x:jnp.ndarray, mean:jnp.ndarray = None,
                     cov:jnp.ndarray=None)->jnp.ndarray
        Computes the density of a multivariate normal evaluated at x
        
    sim_uniform(a:float = 0.0, b:float = 1.0, 
                dim:List[int]=1)->jnp.ndarray
        Simulates uniformly distributed variables
        
    sim_sde(x0:jnp.ndarray, 
            f:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
            g:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
            Wt:jnp.ndarray = None,
            theta:jnp.ndarray = None, #Parameters of the model
            n_sim:int = 10, 
            n_steps:int=100, 
            t0:float = 0.0,
            T:float = 1.0)->Tuple[jnp.ndarray, jnp.ndarray]
        Simulates an Ito Process
        
    ito_integral(Xt:jnp.ndarray,
                n_sim_int:int = 10,
                t0:float=0.0,
                T:float=1.0)->jnp.ndarray
        Estimates the Ito integral
        
    stratonovich_integral(Xt:jnp.ndarray,
                          n_sim_int:int = 10,
                          t0:float=0.0,
                          T:float=1.0)->jnp.ndarray:
        Estimates the Stratonovich Integral
        
    ri_trapez(t:jnp.ndarray, h:float)->jnp.ndarray
        Estimates the Riemennian integral using the trapez method
        
    hessian(fun:Callable[[jnp.ndarray], jnp.ndarray])->Callable[[jnp.ndarray], jnp.ndarray]
        Computes the hessian of a function
        
    jacobian(fun:Callable[[jnp.ndarray], jnp.ndarray])->Callable[[jnp.ndarray],jnp.ndarray]
        Computes the jacobian of a function
        
    """
    def __init__(self, k:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 grad_k:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
        """
        Parameters
        ----------
        seed : int, optional
            The seed value for random sampling
        """
        
        self.k = k
        self.grad_k = grad_k
    
    def __dH_dpi(self, qi:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
    
        K = jnp.apply_along_axis(self.k, -1, qi-self.q, theta)
            
        return (self.p.T*K).T.sum(axis=0)
        
    def __dH_dqi(self, xi:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
        
        N = len(xi)//2
        qi = xi[:N]
        pi = xi[N:]
        
        grad_K = jnp.apply_along_axis(self.grad_k, -1, qi-self.q, theta)
        inner_prod = jnp.dot(self.p, pi)
                        
        return (grad_K.T * inner_prod).T.sum(axis=0)
    
    def __dH_dp(self, theta:jnp.ndarray)->jnp.ndarray:
        
        return jnp.apply_along_axis(self.__dH_dpi, -1, self.q, theta)
    
    def __dH_dq(self, theta:jnp.ndarray)->jnp.ndarray:
        
        x = jnp.hstack((self.q, self.p))
        
        return jnp.apply_along_axis(self.__dH_dqi, -1, x, theta)
    
    def __tv_b_fun(self, t:jnp.ndarray, x:jnp.ndarray, 
                   theta:jnp.ndarray=None)->jnp.ndarray:
        
        x = x.reshape(self.dim)
        q = x[0:self.d]
        p = x[self.d:]
                
        self.q = q
        self.p = p
        dq = self.__dH_dp(theta).reshape(-1)
        dp = -self.__dH_dq(theta).reshape(-1)
        
        return jnp.hstack((dq, dp))
    
    def __tv_sigma_fun(self, t:jnp.ndarray, x:jnp.ndarray, 
                       theta:jnp.ndarray=None)->jnp.ndarray:
        
        val = jnp.diag(self.gamma)
        zero = jnp.zeros_like(val)
        
        return jnp.vstack((zero, val))
    
    def get_tv_model(self, gamma:jnp.ndarray, dim:List[int]
                     )->Tuple[Callable, Callable]:
        
        self.d = dim[0]//2
        self.gamma = jnp.repeat(gamma,dim[1], axis=0)
        self.dim = dim
        
        return self.__tv_b_fun, self.__tv_sigma_fun
    
    def __ms_b_fun(self, t:jnp.ndarray, x:jnp.ndarray,
                   theta:jnp.ndarray=None)->jnp.ndarray:
        
        x = x.reshape(self.dim)
        q = x[0:self.d]
        p = x[self.d:]
                        
        self.q = q
        self.p = p
        dq = self.__dH_dp(theta).reshape(-1)
        dp = -(self.lambda_*self.__dH_dq(theta).reshape(-1)+dq)
        
        return jnp.hstack((dq, dp))
    
    def __ms_sigma_fun(self, t:jnp.ndarray, x:jnp.ndarray,
                       theta:jnp.ndarray=None)->jnp.ndarray:
        
        val = jnp.diag(self.gamma)
        zero = jnp.zeros_like(val)
        
        return jnp.vstack((zero, val))
    
    def get_ms_model(self, gamma:jnp.ndarray, dim:List[int], lambda_:float,
                     )->Tuple[Callable, Callable]:
        
        self.d = dim[0]//2
        self.gamma = jnp.repeat(gamma,dim[1], axis=0)
        self.dim = dim
        self.lambda_ = lambda_
                
        return self.__ms_b_fun, self.__ms_sigma_fun
    
    def __ms_betatilde(self, t:jnp.ndarray, theta:jnp.ndarray=None)->jnp.ndarray:
        
        return jnp.zeros(self.dim[0]*self.dim[1])
    
    def __ms_Btilde(self, t:jnp.ndarray, theta:jnp.ndarray=None)->jnp.ndarray:
        
        K = jnp.apply_along_axis(self.__k_qi, -1, self.qT, theta).reshape(-1,self.dim[0]*self.dim[1]//2)
        K = jnp.vstack((K, -self.lambda_*K))
        zero = jnp.zeros_like(K)
                
        return jnp.hstack((zero, K))
    
    def __ms_sigmatilde_fun(self, t:jnp.ndarray, theta:jnp.ndarray=None)->jnp.ndarray:
        
        val = jnp.diag(self.gamma)
        zero = jnp.zeros_like(val)
        
        return jnp.vstack((zero, val))
    
    def __k_qi(self, qi:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
        
        K_val = jnp.zeros((self.dim[1], self.dim[0]*self.dim[1]//2))
        K = jnp.apply_along_axis(self.k, -1, qi-self.qT, theta).reshape(-1)
        for i in range(self.dim[1]):
            K_val = K_val.at[i,i::self.dim[1]].set(K)
                
        return K_val
    
    def get_ms_approx_model(self, gamma:jnp.ndarray, dim:List[int],
                            qT:jnp.ndarray,
                            lambda_:float,
                            )->Tuple[Callable, Callable, Callable]:
        
        self.gamma = jnp.repeat(gamma,dim[1], axis=0)
        self.dim = dim
        self.qT = qT
        self.lambda_ = lambda_
        
        return self.__ms_betatilde, self.__ms_Btilde, self.__ms_sigmatilde_fun
    
    def landmark_shooting_ivp_rk45(self, x0:jnp.ndarray,
                              k:Callable[[jnp.ndarray], jnp.ndarray],
                              grad_k:Callable[[jnp.ndarray], jnp.ndarray]=None,
                              t0:float=0.0,
                              T:float=1.0,
                              n_steps:int=100
                              )->jnp.ndarray:
        
        dim = list(x0.shape)
        x0 = x0.reshape(-1)
        
        if grad_k is None:
            grad_k = grad(k)
            
        self.k = k
        self.grad_k = grad_k
        self.dim = dim
        self.N = dim[-1]//2
        
        t_eval = jnp.linspace(t0, T, n_steps)
        sol = solve_ivp(self.__landmark_ivp_fun, t_span=[t0, T], y0=x0, 
                        method='RK45', t_eval=t_eval)
        
        y = sol.y
        y = y.reshape(self.dim+[-1])
        qt = y[:,0:self.N,:]
        pt = y[:,self.N:,:]
        
        print(sol.message)
                
        return sol.t, jnp.einsum('ijn->nij', qt), jnp.einsum('ijn->nij', pt)
        
    def __landmark_ivp_fun(self, t:jnp.ndarray, y:jnp.ndarray):
        
        y = y.reshape(self.dim)
        
        self.q = y[:,0:self.N]
        self.p = y[:,self.N:]    
        
        dq = self.__dH_dp()
        dp = -self.__dH_dq()
                
        return jnp.hstack((dq, dp)).reshape(-1)
    
    def landmark_matching_bfgs(self, q0:jnp.ndarray, 
                               qT:jnp.ndarray,
                               k:Callable[[jnp.ndarray], jnp.ndarray],
                               grad_k:Callable[[jnp.ndarray], jnp.ndarray]=None,
                               t0:float=0.0,
                               T:float=1.0,
                               n_steps:int=100,
                               p0:jnp.ndarray=None)->jnp.ndarray:
                
        if p0 is None:
            p0 = jnp.zeros_like(q0)
                
        if grad_k is None:
            grad_k = grad(k)
                
        self.k = k
        self.grad_k = grad_k
        self.t0 = t0
        self.T = T
        self.n_steps = n_steps
        self.q0 = q0
        self.qT = qT
        self.N = qT.shape[-1]
        
        grad_e = grad(self.__landmark_matching_error)
            
        sol = minimize(self.__landmark_matching_error, p0, jac=grad_e,
                 method='BFGS', options={'gtol': 1e-05, 'maxiter':100, 'disp':True})
        
        print(sol.message)
        
        p0 = sol.x.reshape(q0.shape)
    
        x0 = jnp.hstack((q0, p0))
        t, qt, pt = self.landmark_shooting_ivp_rk45(x0, k, grad_k, t0, T, n_steps)
                
        return t, qt, pt
        
    def __landmark_matching_error(self, p0:jnp.ndarray)->jnp.ndarray:
                
        p0 = p0.reshape(self.q0.shape)
        x0 = jnp.hstack((self.q0, p0))
        
        y = x0
        p = p0
        q = self.q0
        dt = (self.T-self.t0)/self.n_steps
        
        for i in range(0,self.n_steps-1):
            self.q = q
            self.p = p
            dq = self.__dH_dp()
            dp = -self.__dH_dq()
            dy = jnp.hstack((dq,dp))
            y += dy*dt
            
            q = y[:, 0:self.N]
            p = y[:, self.N:]
            
        qThat = y[:, 0:self.N]
                            
        return jnp.sum((qThat-self.qT)**2)
    
    def test_approx_ms(self, qT:jnp.ndarray, 
                       k:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                       gamma:jnp.ndarray):
        
        self.k = k
        self.qT = qT
        self.dim = list(qT.shape)
        self.gamma = gamma
        
        return self.__ms_sigmatilde_fun(0,0)

