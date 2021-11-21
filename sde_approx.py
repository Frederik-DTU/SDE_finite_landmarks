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
from jax import random, jacfwd, jacrev, grad

#From scipy
from scipy.integrate import odeint
from scipy.stats import multivariate_normal

#Time modules
import datetime

#From math
import math

#Typing declaration of variables
from typing import Callable, List, Tuple

#%% Class

class sde_finite_landmarks(object):
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
    def __init__(self, seed:int = 2712):
        """
        Parameters
        ----------
        seed : int, optional
            The seed value for random sampling
        """
        
        self.key = random.PRNGKey(seed)
        
    def reset_seed(self, seed:int)->type(None):
        """Updates the seed value

        Parameters
        ----------
        seed : int
            The seed value
        """
        
        self.key = random.PRNGKey(seed)
        
        return
        
    def sim_Wt(self, n_sim:int=1, grid:jnp.ndarray=jnp.linspace(0, 1, 100), 
               dim:int=1)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        """Simulates n_sim for dim dimensional Wiener processes with n_steps
        on the time interval t0 to T on a uniform grid

        Parameters
        ----------
        n_sim : int, optional
            number of realisations
        n_steps : int, optional
            number of steps
        dim : int, optinal
            dimension of the Wiener process
        t0 : float, optional
            start time
        T : float, optional
            End time
            
        Returns
        -------
        tuple of
        -the time grid, t
        -The squeezed wiener process, Wt
        """
        
        n_steps = len(grid)
        N = random.normal(self.key,[n_sim, n_steps-1, dim])
        Wt = jnp.zeros([n_sim, n_steps, dim])
        for i in range(1,n_steps):
            Wt = Wt.at[:,i,:].set(Wt[:,i-1,:]+jnp.sqrt(grid[i]-grid[i-1])*N[:,i-1,:])
            
        return grid, Wt.squeeze()
    
    def sim_dWt(self, n_sim:int=1, grid:jnp.ndarray=jnp.linspace(0, 1, 100), 
               dim:int=1)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        """Simulates n_sim for dim dimensional Wiener processes with n_steps
        on the time interval t0 to T on a uniform grid

        Parameters
        ----------
        n_sim : int, optional
            number of realisations
        n_steps : int, optional
            number of steps
        dim : int, optinal
            dimension of the Wiener process
        t0 : float, optional
            start time
        T : float, optional
            End time
            
        Returns
        -------
        tuple of
        -the time grid, t
        -The squeezed wiener process, Wt
        """
        
        n_steps = len(grid)
        N = random.normal(self.key,[n_sim, n_steps-1, dim])
        dWt = jnp.zeros([n_sim, n_steps, dim])
        for i in range(1,n_steps):
            dWt = dWt.at[:,i,:].set(jnp.sqrt(grid[i]-grid[i-1])*N[:,i-1,:])
            
        return grid, dWt.squeeze()
    
    def sim_multi_normal(self, mu:jnp.ndarray = jnp.zeros(2),
               sigma:jnp.ndarray=jnp.eye(2),
               dim:List[int] = [1])->jnp.ndarray:
        
        """Simulates dim dimensional multivariate normal variables

        Parameters
        ----------
        mu : jnp.ndarray, optional
            mean
        sigma : jnp.ndarray, optional
            covariance matrix
        dim : List[int], optinal
            dimension of the samples
            
        Returns
        -------
        Squeezed normal distributed variables with dimension (dim, mu.shape)
        """
        
        N = random.multivariate_normal(self.key,
                                       mean = mu,
                                       cov=sigma,
                                       shape=dim)
        
        self.key += 1
        
        return N.squeeze()
    
    def multi_normal_pdf(self, x:jnp.ndarray,
                         mean:jnp.ndarray = None,
                         cov:jnp.ndarray=None)->jnp.ndarray:
        
        """Simulates dim dimensional multivariate normal variables

        Parameters
        ----------
        mu : jnp.ndarray
            evaluation point of pdf
        mu : jnp.ndarray, optional
            mean
        sigma : jnp.ndarray, optional
            covariance matrix
            
        Returns
        -------
        Density at x as jnp.ndarray
        """
        
        cov_inv = jnp.linalg.inv(cov)
        x_diff = x-mean
        k = len(x)
        num = jnp.exp(-1/2*(x_diff).T.dot(cov_inv).dot(x_diff))
        den = jnp.sqrt(jnp.linalg.det(cov)*(2*jnp.pi)**k)
        val = num/den
        
        return val
    
    def sim_uniform(self, a:float = 0.0, b:float = 1.0, 
                    dim:List[int]=[1])->jnp.ndarray:
        
        """Simulates dim dimensional uniformly distributed variables

        Parameters
        ----------
        a : float, optional
            start point
        b : float, optional
            end point
        dim : List[int], optinal
            dimension of the samples
            
        Returns
        -------
        Squeezed uniformly distributed variables with dimension dim
        """
        
        U = random.uniform(self.key, shape=dim,
                       minval=a, maxval=b)
        
        self.key += 1
        
        return U
    
    def sim_sde(self, x0:jnp.ndarray, 
                f:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                g:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                Wt:jnp.ndarray = None,
                theta:jnp.ndarray = None, #Parameters of the model
                n_sim:int = 1, 
                grid:jnp.ndarray=jnp.linspace(0, 1, 100)
                )->Tuple[jnp.ndarray, jnp.ndarray]:
        
        """Simulates n_sim realisations of an SDE on an uniform grid using
        the Euler-Marayama method

        Parameters
        ----------
        x0 : jnp.ndarray
            The start point
        f : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
            The drift term given by f(t,Xt,theta)
        g : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
            The diffusion term given by g(t,Xt,theta)
        Wt : jnp.ndarray, optional
            Wiener process of appropiate dimension
        n_sim : int, optional
            number of realisations
        n_steps : int, optional
            number of steps on the uniform time grid
        t0 : float, optional
            start time
        T : float, optional
            end time
            
        Returns
        -------
        tuple of
        -the time grid, t
        -The squeezed Ito process, Xt
        """
        
        dim_brown = g(0,x0, theta).shape[-1]
        n_steps = len(grid)
        sim = jnp.zeros([n_sim, n_steps]+list(x0.shape))
        
        if Wt is None:
            dWt = self.sim_dWt(n_sim=n_sim, grid=grid, dim=dim_brown)
        else:
            dWt = jnp.diff(Wt, axis=0).reshape([n_sim, n_steps-1, dim_brown])

        for i in range(n_sim):
            sim = sim.at[i,0].set(x0)
            for j in range(1,n_steps):
                t_up = grid[j]
                dt = grid[j]-grid[j-1]
                sim = sim.at[i,j].set(sim[i,j-1]+
                                        f(t_up, sim[i,j-1], theta)*dt+
                                           jnp.dot(g(t_up, sim[i,j-1], theta),dWt[i,j-1]))
                
        self.key += 1
        
        return grid, sim.squeeze()
    
    def ito_integral(self, Xt:jnp.ndarray,
                     n_sim_int:int = 10,
                     t0:float=0.0,
                     T:float=1.0)->jnp.ndarray:
        
        """Estimates the Ito integral

        Parameters
        ----------
        Xt : jnp.ndarray
            Stochastic process of dimension (n_sim, n_steps, dim) or (n_steps, dim)
        n_sim_int : int, optional
            Simulations of the Ito integral
        t0 : float, optional
            start time
        T : float, optional
            end time
            
        Returns
        -------
        Estimation of the Ito integral of shape dim
        """
        
        shape = list(Xt.shape)
        lshape = len(shape)
        
        if lshape<=1:
            raise ValueError("Gt must be at least 2 dimensional!")
        elif lshape==2:
            n_sim=1
            n_steps=shape[0]
            dim=shape[1]
        else:
            n_sim=shape[0]
            n_steps=shape[1]
            dim=shape[2]
        
        Xt = Xt.reshape(n_sim, n_steps, dim)
        dt = (T-t0)/n_steps  
        dWt = jnp.sqrt(dt)*random.normal(self.key, [n_sim_int, n_steps-1, dim])

        self.key += 1

        return (Xt[:,:-1,:]*dWt).sum(axis=1).sum(axis=0)/n_sim_int
    
    def stratonovich_integral(self, Xt:jnp.ndarray,
                              n_sim_int:int = 10,
                              t0:float=0.0,
                              T:float=1.0)->jnp.ndarray:
        
        """Estimates the Stratonovich Integral

        Parameters
        ----------
        Xt : jnp.ndarray
            Stochastic process of dimension (n_sim, n_steps, dim) or (n_steps, dim)
        n_sim_int : int, optional
            Simulations of the Ito integral
        t0 : float, optional
            start time
        T : float, optional
            end time
            
        Returns
        -------
        Estimation of the Stratonovich integral of shape dim
        """
        
        shape = list(Xt.shape)
        lshape = len(shape)
        
        if lshape<=1:
            raise ValueError("Gt must be at least 2 dimensional!")
        elif lshape==2:
            n_sim=1
            n_steps=shape[1]
            dim=shape[2]
        else:
            n_sim=shape[0]
            n_steps=shape[1]
            dim=shape[2]
        
        Xt = Xt.reshape(n_sim, n_steps, dim)
        dt = (T-t0)/n_steps 
        dWt = jnp.sqrt(dt)*random.normal(self.key, [n_sim_int, n_steps-1, dim])
        
        self.key += 1
                                
        return ((Xt[:,:-1,:]+Xt[:,1:,:])*dWt).sum(axis=1).sum(axis=0)/(2*n_sim_int)
    
    def ri_trapez(self, ft:jnp.ndarray,
                      grid:jnp.ndarray)->jnp.ndarray:
        
        """Estimates the Riemannian Integral using trapez method

        Parameters
        ----------
        ft : jnp.ndarray
            evaluation of the function
        h : float
            step size
            
        Returns
        -------
        Estimation of the Riemannian integral
        """
        
        n = len(grid)
        ri = 0.0
        for i in range(n):
            ri += (ft[i]+ft[i+1])*(grid[i+1]-grid[i])/2
            
        return ri
    
    def hessian(self, fun:Callable[[jnp.ndarray], jnp.ndarray]
                )->Callable[[jnp.ndarray], jnp.ndarray]:
        
        """Computes the Hessian matrix using autodifferentation

        Parameters
        ----------
        fun : Callable[[jnp.ndarray], jnp.ndarray]
            function of the form f(x)
            
        Returns
        -------
        Hessian
        """
        
        return jacfwd(jacrev((fun)))
    
    def jacobian(self, fun:Callable[[jnp.ndarray], jnp.ndarray]
                 )->Callable[[jnp.ndarray],jnp.ndarray]:
        
        """Computes the Jacobian matrix using autodifferentation

        Parameters
        ----------
        fun : Callable[[jnp.ndarray], jnp.ndarray]
            function of the form f(X)
            
        Returns
        -------
        Jacobian
        """
        
        return jacfwd(fun)
    
    def approx_p0(self, q0:jnp.ndarray,
                  p0:jnp.ndarray,
                  vT:jnp.ndarray,
                  SigmaT:jnp.ndarray,
                  LT:jnp.ndarray,
                  time_grid:jnp.ndarray,
                  b_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigma_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  betatilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  Btilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigmatilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  pi_prob:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  theta:jnp.ndarray = None,
                  max_iter:int = 100,
                  eta:float=0.5,
                  delta:float=0.1,
                  save_path:str='',
                  save_hours:float=1.0,
                  )->jnp.ndarray:
        
        if len(q0.shape)==1:
            q0 = q0.reshape(-1,1)
            
        if len(p0.shape)==1:
            p0 = p0.reshape(-1,1)
        
        x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
        
        a_fun = lambda t,x,par: self.sigma_fun(t, x, par).dot(\
                                self.sigma_fun(t, x, par).transpose())
        g_fun = lambda t,x,par: self.sigma_fun(t,x,par)
        
        self.q0 = q0
        self.vT = vT
        self.SigmaT = SigmaT
        self.LT = LT
        self.time_grid = time_grid
        self.reverse_time_grid = time_grid[::-1] #Reverse order
        self.b_fun = b_fun
        self.sigma_fun = sigma_fun
        self.betatilde_fun = betatilde_fun
        self.Btilde_fun = Btilde_fun
        self.sigmatilde_fun = sigmatilde_fun
        self.pi_prob = pi_prob
        self.max_iter = max_iter 
        self.eta = eta
        self.sqrt_eta = jnp.sqrt(1-eta**2)
        self.delta = delta
        self.sqrt_delta = jnp.sqrt(delta)
        self.dim = len(x0)
        self.dim_brown = sigma_fun(0,x0, theta).shape[-1]
        self.a_fun = a_fun
        self.g_fun = g_fun
        self.dn = len(q0.reshape(-1))
        self.sigmatilde = jnp.apply_along_axis(self.sigmatilde_fun, 1, 
                                         time_grid.reshape(-1,1), theta)
        self.atilde = jnp.matmul(self.sigmatilde, self.sigmatilde.transpose(0,2,1))
        self.betatilde = jnp.apply_along_axis(self.betatilde_fun, 1, 
                                         time_grid.reshape(-1,1), theta)
        self.Btilde = jnp.apply_along_axis(self.Btilde_fun, 1, 
                                         time_grid.reshape(-1,1), theta)
        self.save_path = save_path
        self.save_hours = save_hours
        
        Lt, Mt, mut = self.__solve_backward(theta)
        Mt_inv = jnp.linalg.inv(Mt)
        Ft = self.__compute_Ft(Lt, Mt_inv, self.vT, mut)
        Ht = self.__compute_Ht(Lt, Mt_inv)
        rtilde = lambda t,x, t_vec=time_grid: self.__compute_rtilde(x, Ht[jnp.argmin(t_vec-t)], 
                                             Ft[jnp.argmin(t_vec-t)])
        f_fun = lambda t,x,par: self.b_fun(t,x,par)+\
                                    self.a_fun(t,x,par).dot(rtilde(t,x))
        self.f_fun = f_fun
        self.Ht = Ht
        self.Ft = Ft
        self.mut = mut
        self.Lt = Lt
        self.Mt = Mt
        self.Mt_inv = Mt_inv
        
        _, Wt = self.sim_Wt(n_sim=1, grid=time_grid, dim=self.dim_brown)
        
        _, Xt = self.sim_sde(x0, self.f_fun, self.g_fun, Wt = Wt, 
                                  theta=theta, 
                                  grid=self.time_grid)
        
        self.psi_Xt = self.__compute_psi(Xt, self.Ht, self.Ft, theta)
        self.rho_x0 = self.__compute_rhox0(x0, x0, self.mut[0], self.Mt[0], self.Lt[0])
        self.pi_x0 = pi_prob(q0, p0)
        
        Wt, Xt = self.__approx_p0(Xt, Wt, x0, p0)
            
        return Wt, Xt
    
    def approx_landmark_sde(self, q0:jnp.ndarray,
                            p0:jnp.ndarray,
                            vT:jnp.ndarray,
                            SigmaT:jnp.ndarray,
                            LT:jnp.ndarray,
                            b_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                            sigma_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                            betatilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                            Btilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                            sigmatilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                            pi_prob:Callable[[jnp.ndarray], jnp.ndarray],
                            q_theta:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                            q_sample_theta:Callable[[jnp.ndarray], jnp.ndarray],
                            theta:jnp.ndarray = None,
                            max_iter:int = 100,
                            eta:float=0.5,
                            delta:float=0.01,
                            n_steps:int = 100,
                            t0:float=0.0,
                            T:float=1.0,
                            save_path:str='',
                            save_hours:float=1.0,
                            )->Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        if len(q0.shape)==1:
            q0 = q0.reshape(-1,1)
            
        if len(p0.shape)==1:
            p0 = p0.reshape(-1,1)
        
        x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
        a_fun = lambda t,x,par: self.sigma_fun(t, x, par).dot(\
                                self.sigma_fun(t, x, par).transpose())
        g_fun = lambda t,x,par: self.sigma_fun(t,x,par)
        
        self.q0 = q0
        self.vT = vT
        self.SigmaT = SigmaT
        self.LT = LT
        self.b_fun = b_fun
        self.sigma_fun = sigma_fun
        self.betatilde_fun = betatilde_fun
        self.Btilde_fun = Btilde_fun
        self.sigmatilde_fun = sigmatilde_fun
        self.pi_prob = pi_prob
        self.q_theta = q_theta
        self.q_sample_theta = q_sample_theta
        self.max_iter = max_iter 
        self.eta = eta
        self.sqrt_eta = jnp.sqrt(1-eta**2)
        self.delta = delta
        self.sqrt_delta = jnp.sqrt(delta)
        self.n_steps = n_steps
        self.t0 = t0
        self.T = T
        self.dim = len(x0)
        self.dim_brown = sigma_fun(0,x0, theta).shape[-1]
        self.a_fun = a_fun
        self.g_fun = g_fun
        self.dn = len(q0.reshape(-1))
        self.save_path = save_path
        self.save_hours = save_hours
        
        t_vec, Wt = self.sim_Wt(n_sim=1, n_steps=n_steps, dim=self.dim_brown, 
                            t0=t0, T=T)
        
        self.sigmatilde = jnp.apply_along_axis(self.sigmatilde_fun, 1, 
                                         t_vec.reshape(-1,1), theta)
        self.atilde = jnp.matmul(self.sigmatilde, self.sigmatilde.transpose(0,2,1))
        self.betatilde = jnp.apply_along_axis(self.betatilde_fun, 1, 
                                         t_vec.reshape(-1,1), theta)
        self.Btilde = jnp.apply_along_axis(self.Btilde_fun, 1, 
                                         t_vec.reshape(-1,1), theta)
        
        Lt, Mt, mut = self.__solve_backward(theta)
        Mt_inv = jnp.linalg.inv(Mt)
        Ft = self.__compute_Ft(Lt, Mt_inv, self.vT, mut)
        Ht = self.__compute_Ht(Lt, Mt_inv)
        rtilde = lambda t,x, t_vec=t_vec: self.__compute_rtilde(x, Ht[jnp.argmin(t_vec-t)], 
                                             Ft[jnp.argmin(t_vec-t)])
        f_fun = lambda t,x,par: self.b_fun(t,x,par)+\
                                    self.a_fun(t,x,par).dot(rtilde(t,x))
        self.f_fun = f_fun
        self.Ht = Ht
        self.Ft = Ft
        self.mut = mut
        self.Lt = Lt
        self.Mt = Mt
        self.Mt_inv = Mt_inv
        
        t_vec, Xt = self.sim_sde(x0, self.f_fun, self.g_fun, Wt = Wt, 
                                  theta=theta, 
                                  n_sim=1, n_steps=self.n_steps,
                                  t0=self.t0, T=self.T)
        
        self.psi_Xt = self.__compute_psi(Xt, t_vec, self.Ht, self.Ft, theta)
        self.rho_x0 = self.__compute_rhox0(x0, x0, self.mut[0], self.Mt[0], self.Lt[0])
        self.pi_x0 = self.pi_prob(Xt[0])
        
        if theta is None:
            Wt, Xt = self.__approx_p0(t_vec, Xt, Wt, x0, p0)
        else:
            Wt, Xt, theta = self.__approx_p0_theta(t_vec, Xt, Wt, x0, p0, 
                                                   theta)
            
        return Wt, Xt, theta
    
    def __approx_p0(self, Xt:jnp.ndarray, Wt:jnp.ndarray,
                    x0:jnp.ndarray, p0:jnp.ndarray
                    )->Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        time_diff = datetime.timedelta(hours=self.save_hours)
        start_time = datetime.datetime.now()
        for i in range(self.max_iter):
            #print("Computing iteration: ", i+1)
            Wt, Xt = self.__al_51(x0, None, self.vT, Xt, Wt)
            p0, Xt = self.__al_52(p0, None, self.vT, Xt, Wt)
            x0 = jnp.hstack((self.q0.reshape(-1), p0.reshape(-1)))
            current_time = datetime.datetime.now()
            if current_time - start_time >= time_diff:
                print("Saving iteration: ", i+1)
                save_file = self.save_path+'_'+str(i+1)
                jnp.save(save_file, Xt)
                start_time = current_time
            
        return Wt, Xt
    
    def __approx_p0_theta(self, t_vec:jnp.ndarray, Xt:jnp.ndarray, Wt:jnp.ndarray,
                    x0:jnp.ndarray, p0:jnp.ndarray, theta:jnp.ndarray,
                    )->Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        for i in range(self.max_iter):
            self.__update_matrices(Xt, t_vec, theta)
            print("Computing iteration: ", i+1)
            Wt, Xt = self.__al_51(x0, theta, self.vT, Xt, Wt, t_vec)
            p0, Xt = self.__al_52(p0, theta, self.vT, Xt, Wt)
            x0 = jnp.hstack((self.q0.reshape(-1), p0.reshape(-1)))
            theta, Xt = self.__al_53(x0, theta, Xt, Wt, self.vT)
            
        return Wt, Xt, theta
    
    def __update_matrices(self, Xt:jnp.ndarray, theta:jnp.ndarray)->type(None):
        
        Lt, Mt, mut = self.__solve_backward(theta)
        Mt_inv = jnp.linalg.inv(Mt)
        Ft = self.__compute_Ft(Lt, Mt_inv, self.vT, mut)
        Ht = self.__compute_Ht(Lt, Mt_inv)
        rtilde = lambda t,x, t_vec=self.time_grid: self.__compute_rtilde(x, Ht[jnp.argmin(t_vec-t)], 
                                             Ft[jnp.argmin(t_vec-t)])
        f_fun = lambda t,x,par: self.b_fun(t,x,par)+\
                                    self.a_fun(t,x,par).dot(rtilde(t,x))
        self.f_fun = f_fun
        self.Ht = Ht
        self.Ft = Ft
        self.mut = mut
        self.Lt = Lt
        self.Mt = Mt
        self.Mt_inv = Mt_inv
                
        return
    
    def __al_51(self, x0:jnp.ndarray,
                theta:jnp.ndarray,
                vT:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        U = self.sim_uniform()
        _, Zt = self.sim_Wt(n_sim=1, grid=self.time_grid, dim=self.dim_brown)
        
        Wt_circ = self.eta*Wt+self.sqrt_eta*Zt
        
        _, Xt_circ = self.sim_sde(x0, self.f_fun, self.g_fun, Wt = Wt_circ, 
                                  theta=theta, 
                                  grid=self.time_grid)
        
        psi_Xtcirc = self.__compute_psi(Xt_circ, self.Ht, self.Ft, theta)
        A = jnp.exp(psi_Xtcirc-self.psi_Xt)
        
        if U<A:
            Xt = Xt_circ
            Wt = Wt_circ
            self.psi_Xt = psi_Xtcirc
            
        return Wt, Xt
    
    def __al_52(self, p0:jnp.ndarray,
                theta:jnp.ndarray,
                vT:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        U = self.sim_uniform()
        Z = self.sim_multi_normal(mu=jnp.zeros(self.dn), sigma=jnp.eye(self.dn)).reshape(p0.shape)
                
        L_theta = lambda p0, q0=self.q0, theta=theta, Wt=Wt, \
            Xt=Xt: self.__compute_Ltheta(jnp.hstack((self.q0,p0)).reshape(-1), theta, Wt, Xt)
        grad_L = grad(L_theta)
        L_p0 = grad_L(p0)
        
        p0_circ = p0+self.delta/2*grad_L(p0)+self.sqrt_delta*Z
        x0_circ = jnp.hstack((self.q0.reshape(-1), p0_circ.reshape(-1)))

        _, Xt_circ = self.sim_sde(x0_circ, self.f_fun, self.g_fun, Wt=Wt,
                                  theta=theta, n_sim=1, 
                                  grid=self.time_grid)
        
        L_p0_circ = grad_L(p0_circ)
        
        psi_Xtcirc = self.__compute_psi(Xt_circ, self.Ht, self.Ft, theta)
        rho_x0circ = self.__compute_rhox0(x0_circ, x0_circ, self.mut[0], 
                                          self.Mt[0], self.Lt[0])
        
        pi_x0circ = self.pi_prob(self.q0, p0_circ)
        norm_p0 = multivariate_normal.pdf(p0.reshape(-1),
                                          mean=(p0_circ+self.delta*L_p0_circ/2).reshape(-1), 
                                          cov=self.delta*jnp.eye(len(p0_circ)))
        norm_p0_circ = multivariate_normal.pdf(p0_circ.reshape(-1), 
                                               mean=(p0+self.delta*L_p0/2).reshape(-1), 
                                               cov=self.delta*jnp.eye(len(p0)))

        A = ((rho_x0circ*pi_x0circ*norm_p0_circ)/(self.rho_x0*
                                                        self.pi_x0*
                                                        norm_p0))
        A *= jnp.exp(psi_Xtcirc-self.psi_Xt)
                
        if U<A:
            Xt = Xt_circ
            p0 = p0_circ
            self.psi_Xt = psi_Xtcirc
            self.rho_x0 = rho_x0circ
            self.pi_x0 = pi_x0circ
        
        return p0, Xt
    
    def __al_53(self, x0:jnp.ndarray,
                theta:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray,
                vT:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        theta_circ = self.q_sample(theta)
        
        U = self.sim_uniform(n_sim=1, n_steps=1, dim=1).squeeze()
        
        t_vec, Xt_circ = self.sim_sde(x0, self.f_fun, self.g_fun, theta_circ, n_sim=1, 
                                  Wt = Wt, grid=self.time_grid)
        
        Lt_circ, Mt_circ, mut_circ = self.solve_backward(theta_circ)
        Mt_inv_circ = jnp.linalg.inv(Mt_circ)
        Ft_circ = self.__compute_Ft(Lt_circ, Mt_inv_circ, vT, mut_circ)
        Ht_circ = self.__compute_Ht(Lt_circ, Mt_inv_circ)
        psi_Xtcirc = self.__compute_psi(Xt_circ, t_vec, Ht_circ, Ft_circ)
        rho_x0circ = self.__compute_rhox0(x0, mut_circ[0], Mt_circ[0], Lt_circ[0])
        
        q_theta = self.q(theta, theta_circ)
        q_theta_circ = self.q(theta_circ, theta)
        
        A = (rho_x0circ*q_theta)/(self.rho_x0*q_theta_circ)
        A *= jnp.exp(psi_Xtcirc-self.psi_Xt)
        
        if U<A:
            Xt =Xt_circ
            theta = theta_circ
            self.psi_Xt = psi_Xtcirc
            self.rho_x0 = rho_x0circ
            self.__update_matrices(Xt, t_vec, theta)
            
        return theta, Xt
    
        
    def __compute_Ltheta(self, x0:jnp.ndarray,
                         theta:jnp.ndarray,
                         Wt:jnp.ndarray,
                         Xt:jnp.ndarray)->jnp.ndarray:
        
        val1 = jnp.log(self.psi_Xt)
        val2 = jnp.log(self.__compute_rhox0(x0, x0, 
                                            self.mut[0], self.Mt[0], self.Lt[0]))
        
        return val1+val2
        
        
    def __compute_psi(self, Xt:jnp.ndarray,
                    Ht:jnp.ndarray, #t
                    Ft:jnp.ndarray, #t
                    theta:jnp.ndarray,
                    )->jnp.ndarray:
        
        n_steps = len(self.time_grid)
        sigma = self.sigma_fun(0,Xt[0],theta)
        sigma = jnp.zeros((n_steps, sigma.shape[0], sigma.shape[1]))
        for i in range(n_steps):
            sigma = sigma.at[i].set(self.sigma_fun(self.time_grid[i], Xt[i], theta))
            
        b = jnp.zeros((n_steps, Xt.shape[-1]))
        for i in range(n_steps):
            b = b.at[i].set(self.b_fun(self.time_grid[i], Xt[i], theta))
        
        a = jnp.matmul(sigma, sigma.transpose(0,2,1))
        
        btilde = self.betatilde+jnp.einsum('ijn,in->ij', self.Btilde, Xt)
        rtilde = self.__compute_rtildemat(Xt, Ht, Ft)[...,jnp.newaxis]
                
        val = Ht-jnp.matmul(rtilde, rtilde.transpose(0,2,1))
        
        num = jnp.einsum('ij,ij->i', b-btilde, rtilde.squeeze())
        den = jnp.matmul(a-self.atilde, Ht-val)
        
        Gx = num-1/2*jnp.trace(den,axis1=1, axis2=2)
                
        return self.ri_trapez(Gx,self.time_grid)
    
    def __solve_backward(self, theta:jnp.ndarray
                         )->Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        atilde_fun = lambda t, par=theta: self.sigmatilde_fun(t, par).dot(self.sigmatilde_fun(t, par).transpose())
        betatilde_fun = lambda t, par=theta: self.betatilde_fun(t, par)
        Btilde_fun = lambda t, par=theta: self.Btilde_fun(t, par)
        
        t = self.reverse_time_grid
        LT_dim = list(self.LT.shape)
        LT_dim_flatten = math.prod(LT_dim)
        SigmaT_dim = list(self.SigmaT.shape)
                
        if len(LT_dim)==1:
            muT=jnp.zeros(1)
        else:
            muT = jnp.zeros(LT_dim[0])
            
        y0 = jnp.hstack((self.LT.reshape(-1), self.SigmaT.reshape(-1), 
                         muT))
                
        y = odeint(self.__backward_fun, y0, t, args=(betatilde_fun,
                                                 Btilde_fun,
                                                 atilde_fun,
                                                 LT_dim,
                                                 LT_dim_flatten))
        
        y = y[::-1] #Reversing the array such that yT=yT
        
        Lt = y[:, 0:LT_dim_flatten]
        Mt = y[:, LT_dim_flatten:(LT_dim_flatten+math.prod(SigmaT_dim))]
        mut = y[:,(LT_dim_flatten+math.prod(SigmaT_dim)):]
                
        Lt = Lt.reshape([-1]+LT_dim)
        Mt = Mt.reshape([-1]+SigmaT_dim)
        
        return Lt, Mt, mut
    
    def __backward_fun(self, y:jnp.ndarray, 
                       t:jnp.ndarray,
                       betatilde_fun: Callable[[jnp.ndarray], jnp.ndarray],
                       Btilde_fun:Callable[[jnp.ndarray], jnp.ndarray],
                       atilde_fun:Callable[[jnp.ndarray], jnp.ndarray],
                       LT_dim:List[int],
                       LT_dim_flatten:int):
        
        Lt = y[0:LT_dim_flatten].reshape(LT_dim)
        
        dLt = -jnp.dot(Lt, Btilde_fun(t)) #dL(t)=-L(t)B(t)dt
        dM = -jnp.dot(jnp.dot(Lt, atilde_fun(t)), Lt.transpose()) #dM(t)=-L(t)a(t)L(t)'dt
        dmu = -jnp.dot(Lt, betatilde_fun(t))
        
        rhs = jnp.hstack((dLt.reshape(-1), dM.reshape(-1), dmu.reshape(-1)))
                        
        return rhs
    
    def __compute_Ft(self, Lt:jnp.ndarray, 
                   Mt_inv:jnp.ndarray,
                   vT:jnp.ndarray, #Final observation (qT in landmarks)
                   mut:jnp.ndarray)->jnp.ndarray:
        
        val = jnp.einsum('ijn,ink->ijk', Lt.transpose(0,2,1), Mt_inv)
        
        return jnp.einsum('inj,ij->in', val, vT.reshape(-1)-mut)
    
    def __compute_Ht(self, Lt:jnp.ndarray, 
                   Mt_inv:jnp.ndarray)->jnp.ndarray:
        
        val = jnp.einsum('ijn,ink->ijk', Lt.transpose(0,2,1), Mt_inv)
        
        return jnp.einsum('ijn,ink->ijk', val, Lt)
    
    def __compute_rtilde(self, xt:jnp.ndarray,
                         Ht:jnp.ndarray, 
                         Ft:jnp.ndarray)->jnp.ndarray:
        
        return Ft-jnp.einsum('jk,k->j', Ht, xt)
    
    def __compute_rtildemat(self, Xt:jnp.ndarray,
                         Ht:jnp.ndarray, 
                         Ft:jnp.ndarray)->jnp.ndarray:
        
        return Ft-jnp.einsum('ijk,ik->ij', Ht, Xt)
    
    def __compute_rhotilde(self, Vt:jnp.ndarray,
                         xt:jnp.ndarray,
                         mut:jnp.ndarray,
                         Mt:jnp.ndarray,
                         Lt:jnp.ndarray)->jnp.ndarray:
        
        mean = mut+jnp.einsum('jk,k->j', Lt, xt)
                
        return multivariate_normal.pdf(Vt, mean=mean, cov=Mt)
    
    def __compute_rhox0(self, v0:jnp.ndarray,
                        x0:jnp.ndarray,
                        mu:jnp.ndarray,
                        M0:jnp.ndarray,
                        L0:jnp.ndarray)->jnp.ndarray:
        
        v = jnp.einsum('jk,k->j', L0, x0)
        mean = mu+v
        
        #return self.multi_normal_pdf(v0, mean=mean, cov=M0)
        return self.multi_normal_pdf(v, mean=mean, cov=M0)



