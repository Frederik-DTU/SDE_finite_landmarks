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
        k : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
            Scalar kernel function, k(x,theta)
        grad_k : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
            Gradient of the scalar kernel function, grad_k(x,theta)
        """
        
        self.k = k
        self.grad_k = grad_k
    
    def __dH_dpi(self, qi:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
        
        """Computes dH/dpi = \sum_{j=1}^{n}p[j]k(q[i]-q[j]) for given i

        Parameters
        ----------
        qi : jnp.ndarray
            Landmarks of dimension nxd
        theta : jnp.ndarray
            Parameters of the kernel
            
        Returns
        -------
        jnp.ndarray of dH/dpi = \sum_{j=1}^{n}p[j]k(q[i]-q[j]) for given i
        """
    
        K = jnp.apply_along_axis(self.k, -1, qi-self.q, theta)
            
        return (self.p.T*K).sum(axis=1)
        
    def __dH_dqi(self, xi:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
        
        """Computes dH/dqi = -\sum_{j=1}^{n}<p[i],p[j]>grad_k(q[i]-q[j]) for given i

        Parameters
        ----------
        xi : jnp.ndarray
            (Landmarks, Momentum) of dimension (2*n)xd
        theta : jnp.ndarray
            Parameters of the kernel
            
        Returns
        -------
        jnp.ndarray of dH/dqi = -\sum_{j=1}^{n}<p[i],p[j]>grad_k(q[i]-q[j]) for given i
        """
        
        n = len(xi)//2
        qi = xi[:n]
        pi = xi[n:]
        
        grad_K = jnp.apply_along_axis(self.grad_k, -1, qi-self.q, theta)
        inner_prod = jnp.dot(self.p, pi)
                                
        return (grad_K.T * inner_prod).sum(axis=1)
    
    def __dH_dp(self, theta:jnp.ndarray)->jnp.ndarray:
        
        """Computes dH/dpi = \sum_{j=1}^{n}p[j]k(q[i]-q[j]) for all i

        Parameters
        ----------
        theta : jnp.ndarray
            Parameters of the kernel
            
        Returns
        -------
        jnp.ndarray of dH/dpi = \sum_{j=1}^{n}p[j]k(q[i]-q[j]) for all i
        """
        
        return jnp.apply_along_axis(self.__dH_dpi, -1, self.q, theta)
    
    def __dH_dq(self, theta:jnp.ndarray)->jnp.ndarray:
        
        """Computes dH/dqi = -\sum_{j=1}^{n}<p[i],p[j]>grad_k(q[i]-q[j]) for all i

        Parameters
        ----------
        theta : jnp.ndarray
            Parameters of the kernel
            
        Returns
        -------
        jnp.ndarray of dH/dqi = -\sum_{j=1}^{n}<p[i],p[j]>grad_k(q[i]-q[j]) for all i
        """
        
        x = jnp.hstack((self.q, self.p))
        
        return jnp.apply_along_axis(self.__dH_dqi, -1, x, theta)
    
    def __tv_b_fun(self, t:jnp.ndarray, x:jnp.ndarray, 
                   theta:jnp.ndarray=None)->jnp.ndarray:
        
        """Computes drift of tv-model [dq/dt, dp/dt] = [dH/dp, -dH/dp] 

        Parameters
        ----------
        t : jnp.ndarray
            time
        x : jnp.ndarray
            states
        theta : jnp.ndarray
            Parameters of the kernel
            
        Returns
        -------
        jnp.ndarray: [dq/dt, dp/dt] = [dH/dp, -dH/dp] 
        """
        
        x = x.reshape(self.dim)
        q = x[0:self.n]
        p = x[self.n:]
                
        self.q = q
        self.p = p
        dq = self.__dH_dp(theta).reshape(-1)
        dp = -self.__dH_dq(theta).reshape(-1)
        
        return jnp.hstack((dq, dp))
    
    def __tv_sigma_fun(self, t:jnp.ndarray, x:jnp.ndarray, 
                       theta:jnp.ndarray=None)->jnp.ndarray:
        
        """Computes diffusion of tv-model [0, jnp.eye(gamma)]

        Parameters
        ----------
        t : jnp.ndarray
            time
        x : jnp.ndarray
            states
        theta : jnp.ndarray
            Parameters of the kernel
            
        Returns
        -------
        jnp.ndarray: [0, jnp.eye(gamma)]
        """
        
        val = jnp.diag(self.gamma)
        zero = jnp.zeros_like(val)
        
        return jnp.vstack((zero, val))
    
    def get_tv_model(self, gamma:jnp.ndarray, dim:List[int]
                     )->Tuple[Callable, Callable]:
        
        """Returns the diffusion and drift term in the tv-model

        Parameters
        ----------
        gamma : jnp.ndarray
            coefficients
        dim : List[int]
            dimension of landmarks [2n, d]
            
        Returns
        -------
        Tuple of callables:
            -Drift term in tv-model
            -Diffusion term in tv-model
        """
        
        self.n = dim[0]//2
        self.gamma = jnp.repeat(gamma,dim[1], axis=0)
        self.dim = dim
        
        return self.__tv_b_fun, self.__tv_sigma_fun
    
    def __ms_b_fun(self, t:jnp.ndarray, x:jnp.ndarray,
                   theta:jnp.ndarray=None)->jnp.ndarray:
        
        """Computes drift of ms-model [dq/dt, dp/dt] = [dH/dp, -lambda*dH/dp-dH/dq] 

        Parameters
        ----------
        t : jnp.ndarray
            time
        x : jnp.ndarray
            states
        theta : jnp.ndarray
            Parameters of the kernel
            
        Returns
        -------
        jnp.ndarray: [dq/dt, dp/dt] = [dH/dp, -lambda*dH/dp-dH/dq] 
        """
        
        x = x.reshape(self.dim)
        q = x[0:self.d]
        p = x[self.d:]
                        
        self.q = q
        self.p = p
        dq = self.__dH_dp(theta).reshape(-1)
        dp = -(self.lambda_*dq+(self.__dH_dq(theta)).reshape(-1))
                
        return jnp.hstack((dq, dp))
    
    def __ms_sigma_fun(self, t:jnp.ndarray, x:jnp.ndarray,
                       theta:jnp.ndarray=None)->jnp.ndarray:
        
        """Computes diffusion of ms-model [0, jnp.eye(gamma)]

        Parameters
        ----------
        t : jnp.ndarray
            time
        x : jnp.ndarray
            states
        theta : jnp.ndarray
            Parameters of the kernel
            
        Returns
        -------
        jnp.ndarray: [0, jnp.eye(gamma)]
        """
        
        val = jnp.diag(self.gamma)
        zero = jnp.zeros_like(val)
                
        return jnp.vstack((zero, val))
    
    def get_ms_model(self, gamma:jnp.ndarray, dim:List[int], lambda_:float,
                     )->Tuple[Callable, Callable]:
        
        """Returns the diffusion and drift term in the ms-model

        Parameters
        ----------
        gamma : jnp.ndarray
            coefficients
        dim : List[int]
            dimension of landmarks [2n, d]
        lambda_ : float
            Multiplicative term in the ms-model drift
            
        Returns
        -------
        Tuple of callables:
            -Drift term in ms-model
            -Diffusion term in ms-model
        """
        
        self.d = dim[0]//2
        self.gamma = jnp.repeat(gamma,dim[1], axis=0)
        self.dim = dim
        self.lambda_ = lambda_
                
        return self.__ms_b_fun, self.__ms_sigma_fun
    
    def __ms_betatilde(self, t:jnp.ndarray, theta:jnp.ndarray=None)->jnp.ndarray:
        
        """Returns betatilde_fun(t) for the auxiallary process for the ms-model

        Parameters
        ----------
        t : jnp.ndarray
            time
        theta : jnp.ndarray
            parameters of the model
            
        Returns
        -------
        Betatilde_fun(t)
        """
        
        return jnp.zeros(self.dim[0]*self.dim[1])
    
    def __ms_Btilde(self, t:jnp.ndarray, theta:jnp.ndarray=None)->jnp.ndarray:
        
        """Returns Btilde_fun(t) for the auxiallary process for the ms-model

        Parameters
        ----------
        t : jnp.ndarray
            time
        theta : jnp.ndarray
            parameters of the model
            
        Returns
        -------
        Btilde_fun(t)
        """
        
        K = jnp.apply_along_axis(self.__k_qTi, -1, self.qT, theta).\
            reshape(-1,self.dim[0]*self.dim[1]//2)
        
        K = jnp.vstack((K, -self.lambda_*K))
        zero = jnp.zeros_like(K)
                        
        return jnp.hstack((zero, K))
    
    def __k_qTi(self, qTi:jnp.ndarray, theta:jnp.ndarray)->jnp.ndarray:
        
        """Computes k(qTi-qT)

        Parameters
        ----------
        qTi : jnp.ndarray
            The i'th landmark at time T
        theta : jnp.ndarray
            parameters of the model
            
        Returns
        -------
        Array of k(qTi-qT)
        """
        
        K_val = jnp.zeros((self.dim[1], self.dim[0]*self.dim[1]//2))
        K = jnp.apply_along_axis(self.k, -1, qTi-self.qT, theta).reshape(-1)

        for i in range(self.dim[1]):
            K_val = K_val.at[i,i::self.dim[1]].set(K)

        return K_val
    
    def __ms_sigmatilde_fun(self, t:jnp.ndarray, theta:jnp.ndarray=None)->jnp.ndarray:
        
        """Computes sigmatilde_fun of ms-model [0, jnp.eye(gamma)]

        Parameters
        ----------
        t : jnp.ndarray
            time
        theta : jnp.ndarray
            Parameters of the kernel
            
        Returns
        -------
        jnp.ndarray: [0, jnp.eye(gamma)]
        """
        
        val = jnp.diag(self.gamma)
        zero = jnp.zeros_like(val)
        
        return jnp.vstack((zero, val))
    
    def get_ms_approx_model(self, gamma:jnp.ndarray, dim:List[int],
                            qT:jnp.ndarray,
                            lambda_:float,
                            )->Tuple[Callable, Callable, Callable]:
        
        """Returns the diffusion and drift term in the auxilliary ms-model

        Parameters
        ----------
        gamma : jnp.ndarray
            coefficients
        qT : jnp.ndarray
            landmarks at time T
        lambda_ : float
            Multiplicative term in the ms-model drift
            
        Returns
        -------
        Tuple of callables:
            -betatilde in the auxilliary ms-model
            -Btilde in the auxilliary ms-model
            -sigmatilde in the auxilliary ms-model
        """
        
        self.gamma = jnp.repeat(gamma,dim[1], axis=0)
        self.dim = dim
        self.qT = qT
        self.lambda_ = lambda_
        
        return self.__ms_betatilde, self.__ms_Btilde, self.__ms_sigmatilde_fun
    
    def landmark_shooting_ivp_rk45(self, x0:jnp.ndarray,
                                   rhs_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                                   time_grid:jnp.ndarray,
                                   theta:jnp.ndarray=None
                                   )->Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        """Computes (qt,pt) conditioned on (q0,p0)

        Parameters
        ----------
        x0 : jnp.ndarray
            Initial value, (q0,p0)
        rhs : Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
            right hand side of (dq/dt, dp/dt) as flatten array
        time_grid : jnp.ndarray
            time grid to evaluate the solution
        theta : jnp.ndarray
            parameters given to rhs_fun
            
        Returns
        -------
        Tuple of jnp.ndarray:
            -time grid
            -Landmarks qt
            -Momentums pt
        """
        
        dim = list(x0.shape)
        x0 = x0.reshape(-1)
        t0 = time_grid[0]
        T = time_grid[-1]
        
        self.dim = dim
        self.N = dim[-1]//2
        
        sol = solve_ivp(rhs_fun, t_span=[t0, T], y0=x0, 
                        method='RK45', t_eval=time_grid, args=theta)
        
        y = sol.y
        y = y.reshape(self.dim+[-1])
        qt = y[:,0:self.N]
        pt = y[:,self.N:]
        
        print(sol.message)
                
        return sol.t, qt, pt
    
    def landmark_matching_bfgs(self, q0:jnp.ndarray, 
                               qT:jnp.ndarray,
                               rhs_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                               time_grid:jnp.ndarray,
                               theta:jnp.ndarray = None,
                               p0:jnp.ndarray=None)->jnp.ndarray:
        
        """Computes (qt,pt) conditioned on (q0,qT)
        
        Parameters
        ----------
        q0 : jnp.ndarray
            landmarks at time 0
        qT : jnp.ndarray
            landmarks at time T
        rhs : Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
            right hand side of (dq/dt, dp/dt) as flatten array
        time_grid : jnp.ndarray
            time grid to evaluate the solution
        theta : jnp.ndarray
            parameters given to rhs_fun
        p0 : jnp.ndarray
            Initial guess on p0
            
        Returns
        -------
        Tuple of jnp.ndarray:
            -time grid
            -Landmarks qt
            -Momentums pt
        """
        
        self.q0 = q0.reshape(-1)
        self.qT = qT.reshape(-1)
        self.time_grid = time_grid
        self.time_diff = jnp.diff(time_grid, axis=0)
        self.n_steps = len(time_grid)
        self.N = len(self.qT)
        self.rhs_fun = lambda t,x : rhs_fun(t,x,theta)
        self.theta = theta
                
        if p0 is None:
            p0 = jnp.zeros_like(q0)
        
        grad_e = grad(self.__landmark_matching_error)
            
        sol = minimize(self.__landmark_matching_error, p0, jac=grad_e,
                 method='BFGS', options={'gtol': 1e-05, 'maxiter':100, 'disp':True})
        
        print(sol.message)
        
        p0 = sol.x.reshape(q0.shape)
    
        x0 = jnp.hstack((q0, p0))
        t, qt, pt = self.landmark_shooting_ivp_rk45(x0, rhs_fun, time_grid, theta)
                
        return t, qt, pt
        
    def __landmark_matching_error(self, p0:jnp.ndarray)->jnp.ndarray:
        
        """Computes sum (qThat-qT)**2
        
        Parameters
        ----------
        p0 : jnp.ndarray
            momentum at time 0
            
        Returns
        -------
        sum (qThat-qT)**2
        """
                
        x0 = jnp.hstack((self.q0, p0))
        for i in range(0,self.n_steps-1):
            x0 += self.rhs_fun(self.time_grid[i+1], x0)*self.time_diff[i]
            
        qThat = x0[:, 0:self.N]
                            
        return jnp.sum((qThat-self.qT)**2, axis=0)
