#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:55:05 2021

@author: root
"""
#%% Modules used

import jax.numpy as jnp
from jax import vmap, grad, lax

#Typing declaration of variables
from typing import Callable, List, Tuple

#Own modules
import integration_ode as inter
import backward_filtering as bf
import sim_sp as sp

#%% Functions

def landmark_segment(q0:jnp.ndarray,
                  vT:jnp.ndarray,
                  SigmaT:jnp.ndarray,
                  LT:jnp.ndarray,
                  b_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigma_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  beta_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  B_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigmatilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  pi_prob:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  time_grid:jnp.ndarray = jnp.linspace(0,1,100),
                  max_iter:int = 100,
                  eta:float=0.98,
                  delta:float=0.01,
                  theta = None,
                  q_sample = None,
                  q_sample_prob = None,
                  q_prob = None,
                  backward_method = 'odeint',
                  Wt = None,
                  p0:jnp.ndarray = None,
                  save_step = 0,
                  save_path = ''
                  )->Tuple[jnp.ndarray, jnp.ndarray]:
    
    global L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, B_mat, atilde_mat, \
            sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, L_p0, pi_x0
            
    def sim_gp(x0, b_fun, sigma_fun, Ft, Ht, Wt):
        
        def update_xi(xi, ite):
            
            t, dt, dWt, F, H = ite
            sigma = sigma_fun(t,xi)
            a = sigma.dot(sigma.T)
            r = F-H.dot(xi)
            x = xi+(b_fun(t,xi)+a.dot(r))*dt+sigma.dot(dWt)
            
            return x, x
        
        dW = jnp.diff(Wt, axis=0)
        _, x = lax.scan(update_xi, init=x0, xs=(time_grid[:-1], diff_grid, dW, 
                                                Ft[:-1], Ht[:-1]))
        
        return jnp.concatenate((x0.reshape(1,-1), x), axis=0)
    
    def logpsi(b_fun, sigma_fun, beta_mat, B_mat, Ft, Ht, atilde_mat, Xt):
        
        #Consider extending this to 4d such that e.g. beta_fun is a I long list
        b_mat = vmap(b_fun)(time_grid, Xt)
        btilde_mat = beta_mat + jnp.einsum('ijk,ik->ij', B_mat, Xt)
        sigma_mat = vmap(sigma_fun)(time_grid, Xt)
        a_mat = jnp.matmul(sigma_mat, sigma_mat.transpose(0,2,1))
        r_mat = Ft-jnp.einsum('ijk,ik->ij', Ht, Xt)
        
        drift_term = jnp.einsum('ij,ij->i', b_mat-btilde_mat, r_mat)
        Hterm1 = Ht-jnp.einsum('ij,ik->ijk', r_mat,r_mat)
        Hterm2 = a_mat-atilde_mat
        
        Gx = drift_term-1/2*jnp.trace(jnp.matmul(Hterm1,Hterm2),axis1=1, axis2=2)
        
        val, _ = inter.integrator(Gx, time_grid, method='trapez')
        
        return val
    
    def update_mat(theta, Wt):
        
        if theta is not None:
            b_fun_theta = lambda t,x: b_fun(t,x,theta)
            sigma_fun_theta = lambda t,x: sigma_fun(t,x,theta)
            beta_fun_theta = lambda t: beta_fun(t,theta)
            B_fun_theta = lambda t: B_fun(t, theta)
            sigmatilde_fun_theta = lambda t: sigmatilde_fun(t, theta)
        else:
            b_fun_theta = b_fun
            sigma_fun_theta = sigma_fun
            beta_fun_theta = beta_fun
            B_fun_theta = B_fun
            sigmatilde_fun_theta = sigmatilde_fun
            
        atilde_fun = lambda t : jnp.dot(sigmatilde_fun_theta(t), \
                                            sigmatilde_fun_theta(t).T)
            
        beta_mat = vmap(beta_fun_theta)(time_grid)
        B_mat = vmap(B_fun_theta)(time_grid)
        atilde_mat = vmap(atilde_fun)(time_grid)

        L0, M0, mu0, Ft, Ht \
            = bf.lmmu_step(beta_fun_theta, B_fun_theta, 
                           atilde_fun, LT, SigmaT, muT, vT.reshape(-1), time_grid,
                           method=backward_method)
            
        M0_inv = jnp.linalg.inv(M0)
        
        Xt = sim_gp(x0, b_fun_theta, sigma_fun_theta, Ft, Ht, Wt)
        logpsi_Xt = logpsi(b_fun_theta, sigma_fun_theta, beta_mat, B_mat, 
                           Ft, Ht, atilde_mat, Xt)
        rho0 = sp.mnormal_pdf(vT.reshape(-1), mu0+jnp.einsum('jk,k->j', L0, x0), M0)
        
        return L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, \
            B_mat, atilde_mat, sigma_fun_theta, logpsi_Xt, \
                rho0, b_fun_theta, Xt
    
    def update_W_X(x0:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:

        global logpsi_Xt
                
        U = sp.sim_unif()
        
        Zt = sp.sim_Wt(time_grid, dim_brown)
        Wt_circ = eta*Wt+sqrt_eta*Zt
        Xt_circ = sim_gp(x0, b_fun_theta, sigma_fun_theta, Ft, Ht, Wt_circ)
        
        logpsi_Xtcirc = logpsi(b_fun_theta, sigma_fun_theta, beta_mat, B_mat, 
                               Ft, Ht, atilde_mat, Xt_circ)
        
        A = jnp.exp(logpsi_Xtcirc-logpsi_Xt)
        
        if U<A:
            Xt = Xt_circ
            Wt = Wt_circ
            logpsi_Xt = logpsi_Xtcirc
            
        return Wt, Xt
    
    def compute_Ltheta(p0:jnp.ndarray, Wt:jnp.ndarray)->jnp.ndarray:
            
        x0 = jnp.hstack((q0,p0))
        Xt = sim_gp(x0, b_fun_theta, sigma_fun_theta, Ft, Ht, Wt)
        logpsi_Xt = logpsi(b_fun_theta, sigma_fun_theta, beta_mat, B_mat, 
                           Ft, Ht, atilde_mat, Xt)
        
        x_diff = vT.reshape(-1)-(mu0+jnp.einsum('jk,k->j', L0, x0))
        num = -1/2*(x_diff).T.dot(M0_inv).dot(x_diff)
                
        return logpsi_Xt + num
    
    def update_p0_X(p0:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global logpsi_Xt, rho0, pi_x0, L_p0
        
        U = sp.sim_unif()
        Z = sp.sim_multinormal(mu=muT,cov=I)
        
        p0_circ = p0+delta2*L_p0+sqrt_delta*Z
        x0_circ = jnp.hstack((q0, p0_circ))

        Xt_circ = sim_gp(x0_circ, b_fun_theta, sigma_fun_theta, Ft, Ht, Wt)
        logpsi_Xtcirc = logpsi(b_fun_theta, sigma_fun_theta, beta_mat, B_mat, Ft, Ht, 
                               atilde_mat, Xt_circ)
        
        L_p0circ = grad_L(p0_circ, Wt)
        if jnp.linalg.norm(delta2*L_p0circ)>10.0:
            L_p0circ = 0.01*jnp.sign(L_p0circ)/delta2
        
        rho0_circ = sp.mnormal_pdf(vT.reshape(-1),mu0+jnp.einsum('jk,k->j', L0, x0_circ), M0)
        pi_x0circ = pi_prob(q0, p0_circ)
        
        norm_p0 = sp.mnormal_pdf(p0, p0_circ+delta2*L_p0circ, deltaI)
        norm_p0_circ = sp.mnormal_pdf(p0_circ, p0+delta2*L_p0, deltaI)

        A = ((rho0_circ*pi_x0circ*norm_p0)/(rho0*pi_x0*norm_p0_circ))
        A *= jnp.exp(logpsi_Xtcirc-logpsi_Xt)
                                
        if U<A:
            Xt = Xt_circ
            p0 = p0_circ
            logpsi_Xt = logpsi_Xtcirc
            rho0 = rho0_circ
            pi_x0 = pi_x0circ
            L_p0 = L_p0circ
        
        return p0, Xt
    
    def update_theta_X(x0:jnp.ndarray,
                theta:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, B_mat, atilde_mat, \
            f_fun, sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, a_fun
        
        U = sp.sim_unif()
        theta_circ = q_sample(theta)
        
        L0circ, M0circ, M0_invcirc, mu0circ, Htcirc, Ftcirc, beta_matcirc, B_matcirc, \
            atilde_matcirc, sigma_fun_thetacirc, logpsi_Xtcirc, rho0_circ, \
                b_fun_thetacirc, Xtcirc = update_mat(theta_circ, Wt)
        
        pi_theta_circ = q_prob(theta_circ)
        pi_theta = q_prob(theta)
        q_theta_circ = q_sample_prob(theta_circ, theta)
        q_theta = q_sample_prob(theta, theta_circ)
        
        A = (rho0_circ*pi_theta_circ*q_theta)/(rho0*pi_theta*q_theta_circ)
        A *= jnp.exp(logpsi_Xtcirc-logpsi_Xt)
        
        if U<A:
            L0 = L0circ
            M0 = M0circ
            M0_inv = M0_invcirc
            mu0 = mu0circ
            Ht = Htcirc
            Ft = Ftcirc
            beta_mat = beta_matcirc
            B_mat = B_matcirc
            atilde_mat = atilde_matcirc
            sigma_fun_theta = sigma_fun_thetacirc
            logpsi_Xt = logpsi_Xtcirc
            rho0 = rho0_circ
            b_fun_theta = b_fun_thetacirc
            Xt = Xtcirc
            theta = theta_circ
            q_theta = q_theta_circ
            
        return theta, Xt
        
    if theta is not None:
        dim_brown = sigmatilde_fun(time_grid[0], theta).shape[-1]
    else:
        dim_brown = sigmatilde_fun(time_grid[0]).shape[-1]
    if Wt is None:
        Wt = sp.sim_Wt(time_grid, dim_brown)
    if p0 is None:
        p0 = jnp.zeros_like(q0)
        
    p0 = p0.reshape(-1)
    q0 = q0.reshape(-1)
    x0 = jnp.hstack((q0, p0))  
    nd = len(q0)
    
    I = jnp.eye(nd)
    sqrt_eta = jnp.sqrt(1-eta**2)
    sqrt_delta = jnp.sqrt(delta)
    delta2 = delta/2
    deltaI = delta*I
    diff_grid = jnp.diff(time_grid, axis=0)

    muT = jnp.zeros(nd)
    L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, B_mat, atilde_mat, \
            sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, Xt \
                = update_mat(theta, Wt)
                
    grad_L = grad(compute_Ltheta, argnums=0)
    L_p0 = grad_L(p0, Wt)
    if jnp.linalg.norm(L_p0)>1.0:
        L_p0 = 0.01*jnp.sign(L_p0)
    
    pi_x0 = pi_prob(q0,p0)

    if save_step==0:
        if theta is None:
            for i in range(max_iter):
                print("Computing iteration: ", i+1)
                Wt, Xt = update_W_X(x0, Xt, Wt)
                p0, Xt = update_p0_X(p0, Xt, Wt)
                x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
            return Wt, Xt
        else:
            theta_list = []
            theta_list.append(theta)
            for i in range(max_iter):
                print("Computing iteration: ", i+1)   
                Wt, Xt = update_W_X(x0, Xt, Wt)
                p0, Xt = update_p0_X(p0, Xt, Wt)
                x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                theta, Xt = update_theta_X(x0, theta, Xt, Wt)
                theta_list.append(theta)
                
            return theta_list, Wt, Xt
    else:
        if theta is None:
            for i in range(max_iter):
                Wt, Xt = update_W_X(x0, Xt, Wt)
                p0, Xt = update_p0_X(p0, Xt, Wt)
                x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                
                if (i+1) % save_step==0:
                    print("Computing iteration: ", i+1)
                    jnp.savez(save_path+'_iter_'+str(i+1), Wt=Wt, Xt=Xt)
                
            return Wt, Xt
        else:
            theta_list = []
            theta_list.append(theta)
            for i in range(max_iter):
                Wt, Xt = update_W_X(x0, Xt, Wt)
                p0, Xt = update_p0_X(p0, Xt, Wt)
                x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                theta, Xt = update_theta_X(x0, theta, Xt, Wt)
                theta_list.append(theta)
                
                if (i+1) % save_step==0:
                    print("Computing iteration: ", i+1)
                    jnp.savez(save_path+'_iter_'+str(i+1), Wt=Wt, Xt=Xt, 
                              theta=jnp.vstack(theta_list).squeeze())
            
            return jnp.vstack(theta_list).squeeze(), Wt, Xt
    
def landmark_template(vT:jnp.ndarray,
                  SigmaT:jnp.ndarray,
                  LT:jnp.ndarray,
                  b_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigma_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  beta_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  B_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigmatilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  k:Callable[[jnp.ndarray], jnp.ndarray],
                  pi_prob:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  time_grid:jnp.ndarray = jnp.linspace(0,1,100),
                  n = 3,
                  d = 1,
                  max_iter:int = 100,
                  eta:float=0.98,
                  deltaq:float=0.01,
                  deltap:float=0.01,
                  theta = None,
                  q_sample = None,
                  q_sample_prob = None,
                  q_prob = None,
                  backward_method = 'odeint',
                  Wt = None,
                  q0 = None,
                  p0 = None,
                  update_p0 = False,
                  save_step = 0,
                  save_path = ''
                  )->Tuple[jnp.ndarray, jnp.ndarray]:
    
    global L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, B_mat, atilde_mat, \
            sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, L_p0, L_q0, pi_x0, Kq0
            
    def sim_gp(x0, b_fun, sigma_fun, Ft, Ht, Wt):
        
        def update_xi(xi, ite):
            
            t, dt, dWt, F, H = ite
            sigma = sigma_fun(t,xi)
            a = sigma.dot(sigma.T)
            r = F-H.dot(xi)
            x = xi+(b_fun(t,xi)+a.dot(r))*dt+sigma.dot(dWt)
            
            return x, x
        
        dW = jnp.diff(Wt, axis=0)
        _, x = lax.scan(update_xi, init=x0, xs=(time_grid[:-1], diff_grid, dW, 
                                                Ft[:-1], Ht[:-1]))
        
        return jnp.concatenate((x0.reshape(1,-1), x), axis=0)
        
    def logpsi(b_fun, sigma_fun, beta_mat, B_mat, Ft, Ht, atilde_mat, Xt):
        
        b_mat = vmap(b_fun)(time_grid, Xt)
        btilde_mat = beta_mat + jnp.einsum('ijk,ik->ij', B_mat, Xt)
        sigma_mat = vmap(sigma_fun)(time_grid, Xt)
        a_mat = jnp.matmul(sigma_mat, sigma_mat.transpose(0,2,1))
        r_mat = Ft-jnp.einsum('ijk,ik->ij', Ht, Xt)
        
        drift_term = jnp.einsum('ij,ij->i', b_mat-btilde_mat, r_mat)
        Hterm1 = Ht-jnp.einsum('ij,ik->ijk', r_mat,r_mat)
        Hterm2 = a_mat-atilde_mat

        Gx = drift_term-1/2*jnp.trace(jnp.matmul(Hterm1,Hterm2),axis1=1, axis2=2)
        
        val, _ = inter.integrator(Gx, time_grid, method='trapez')
        
        return val
    
    def kernel_matrix(q0):
        
        def compute_row(qi):
            
            return vmap(k)(q0-qi)
        
        q0 = q0.reshape(n,d)
        
        return vmap(compute_row)(q0)

    def update_mat(theta, Wt):
        
        if theta is not None:
            b_fun_theta = lambda t,x: b_fun(t,x,theta)
            sigma_fun_theta = lambda t,x: sigma_fun(t,x,theta)
            beta_fun_theta = lambda t: beta_fun(t,theta)
            B_fun_theta = lambda t: B_fun(t, theta)
            sigmatilde_fun_theta = lambda t: sigmatilde_fun(t, theta)
        else:
            b_fun_theta = b_fun
            sigma_fun_theta = sigma_fun
            beta_fun_theta = beta_fun
            B_fun_theta = B_fun
            sigmatilde_fun_theta = sigmatilde_fun
            
        atilde_fun = lambda t : jnp.dot(sigmatilde_fun_theta(t), \
                                            sigmatilde_fun_theta(t).T)
        beta_mat = vmap(beta_fun_theta)(time_grid)
        B_mat = vmap(B_fun_theta)(time_grid)
        atilde_mat = vmap(atilde_fun)(time_grid)
        
        L0, M0, mu0, Ft, Ht \
            = bf.lmmu_step_template(beta_fun_theta, B_fun_theta, 
                           atilde_fun, LT, SigmaT, muT, vT, time_grid,
                           method=backward_method)
        M0_inv = jnp.linalg.inv(M0)
        
        Xt = vmap(lambda F,w: sim_gp(x0, b_fun_theta, sigma_fun_theta, F, Ht, w))(Ft,Wt)
        logpsi_Xt = vmap(lambda F, x: logpsi(b_fun_theta, sigma_fun_theta, beta_mat, 
                                B_mat, F, Ht, atilde_mat, x))(Ft,Xt)
        rho0 = sp.mnormal_pdf(vT[0], mu0+jnp.einsum('jk,k->j', L0, x0), M0) #Maybe vmap
        
        return L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, \
            B_mat, atilde_mat, sigma_fun_theta, logpsi_Xt, \
                rho0, b_fun_theta, Xt
    
    def update_W_X(x0:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:

        global logpsi_Xt
                
        U = sp.sim_unif(dim=I_obs)
        Zt = sp.sim_Wt(time_grid, dim=dim_brown, simulations=I_obs)
        
        Wt_circ = eta*Wt+sqrt_eta*Zt
        Xt_circ = vmap(lambda F,w: sim_gp(x0, b_fun_theta, 
                                          sigma_fun_theta, F, Ht, w))(Ft,Wt_circ)
        logpsi_Xtcirc = vmap(lambda F, x: logpsi(b_fun_theta, sigma_fun_theta, beta_mat, 
                                B_mat, F, Ht, atilde_mat, x))(Ft,Xt_circ)
        
        A = jnp.exp(logpsi_Xtcirc-logpsi_Xt)
        
        bool_val = U<A
        
        Xt = Xt.at[bool_val].set(Xt_circ[bool_val])
        Wt = Wt.at[bool_val].set(Wt_circ[bool_val])
        logpsi_Xt = logpsi_Xt.at[bool_val].set(logpsi_Xtcirc[bool_val])
            
        return Wt, Xt
    
    def compute_pLtheta(p0:jnp.ndarray, Wt:jnp.ndarray)->jnp.ndarray:
            
        x0 = jnp.hstack((q0,p0))
        
        Xt = vmap(lambda F,w: sim_gp(x0, b_fun_theta, sigma_fun_theta, F, Ht, w))(Ft,Wt)
        logpsi_Xt = vmap(lambda F, x: logpsi(b_fun_theta, sigma_fun_theta, beta_mat, 
                                B_mat, F, Ht, atilde_mat, x))(Ft,Xt)
        
        x_diff = vT[0]-(mu0+jnp.einsum('jk,k->j', L0, x0)) #Maybe vmap
        num = -1/2*(x_diff).T.dot(M0_inv).dot(x_diff)
                
        return jnp.sum(logpsi_Xt) + num
    
    def compute_qLtheta(q0:jnp.ndarray, Wt:jnp.ndarray)->jnp.ndarray:
            
        x0 = jnp.hstack((q0,p0))
        
        Xt = vmap(lambda F,w: sim_gp(x0, b_fun_theta, sigma_fun_theta, F, Ht, w))(Ft,Wt)
        logpsi_Xt = vmap(lambda F, x: logpsi(b_fun_theta, sigma_fun_theta, beta_mat, 
                                B_mat, F, Ht, atilde_mat, x))(Ft,Xt)
        
        x_diff = vT[0]-(mu0+jnp.einsum('jk,k->j', L0, x0)) #Maybe vmap
        num = -1/2*(x_diff).T.dot(M0_inv).dot(x_diff)
                
        return jnp.sum(logpsi_Xt) + num
    
    def update_p0_X(q0,
                    p0:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global logpsi_Xt, rho0, pi_x0, L_p0
        
        U = sp.sim_unif()
        Z = sp.sim_multinormal(mu=muT,cov=I)
        
        p0_circ = p0+deltap2*L_p0+sqrt_deltap*Z
        x0_circ = jnp.hstack((q0, p0_circ))

        Xt_circ = vmap(lambda F,w: sim_gp(x0_circ, b_fun_theta, sigma_fun_theta, F, Ht, w))(Ft,Wt)
        logpsi_Xtcirc = vmap(lambda F, x: logpsi(b_fun_theta, sigma_fun_theta, beta_mat, 
                                B_mat, F, Ht, atilde_mat, x))(Ft,Xt_circ)
        
        L_p0circ = grad_pL(p0_circ, Wt)
        rho0_circ = sp.mnormal_pdf(vT[0],mu0+jnp.einsum('jk,k->j', L0, x0_circ), M0)#Maybe vmap
        pi_x0circ = pi_prob(q0, p0_circ)
        
        norm_p0 = sp.mnormal_pdf(p0, p0_circ+deltap2*L_p0circ, deltapI)
        norm_p0_circ = sp.mnormal_pdf(p0_circ, p0+deltap2*L_p0, deltapI)

        A = ((rho0_circ*pi_x0circ*norm_p0)/(rho0*pi_x0*norm_p0_circ))
        A *= jnp.exp(jnp.sum(logpsi_Xtcirc-logpsi_Xt, axis=0))
        
        if jnp.isnan(A) or jnp.isinf(A):
            A = 1.0
                                
        if U<A:
            Xt = Xt_circ
            p0 = p0_circ
            logpsi_Xt = logpsi_Xtcirc
            rho0 = rho0_circ
            pi_x0 = pi_x0circ
            L_p0 = L_p0circ
        
        return p0, Xt
    
    def update_q0_X(q0:jnp.ndarray,
                    p0,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global logpsi_Xt, rho0, pi_x0, L_q0, Kq0
        
        U = sp.sim_unif()
        Z = sp.sim_multinormal(mu=muT,cov=Kq0)
        
        q0_circ = q0+deltaq2*L_q0+sqrt_deltaq*Z
        x0_circ = jnp.hstack((q0_circ, p0))
        
        Xt_circ = vmap(lambda F,w: sim_gp(x0_circ, b_fun_theta, sigma_fun_theta, F, Ht, w))(Ft,Wt)
        logpsi_Xtcirc = vmap(lambda F, x: logpsi(b_fun_theta, sigma_fun_theta, beta_mat, 
                                B_mat, F, Ht, atilde_mat, x))(Ft,Xt_circ)
        
        L_q0circ = grad_qL(q0_circ, Wt)
        rho0_circ = sp.mnormal_pdf(vT[0],mu0+jnp.einsum('jk,k->j', L0, x0_circ), M0) #Maybe vmap
        pi_x0circ = pi_prob(q0_circ, p0)
        Kq0_circ = kernel_matrix(q0_circ)
        
        norm_q0 = sp.mnormal_pdf(q0, q0_circ+deltaq2*Kq0_circ.dot(L_q0circ), 
                                 deltaq*Kq0_circ)
        norm_q0_circ = sp.mnormal_pdf(q0_circ, q0+deltaq2*Kq0.dot(L_q0), 
                                      deltaq*Kq0)

        A = ((rho0_circ*pi_x0circ*norm_q0)/(rho0*pi_x0*norm_q0_circ))
        A *= jnp.exp(jnp.sum(logpsi_Xtcirc-logpsi_Xt, axis=0))
                                
        if U<A:
            Xt = Xt_circ
            q0 = q0_circ
            logpsi_Xt = logpsi_Xtcirc
            rho0 = rho0_circ
            pi_x0 = pi_x0circ
            L_q0 = L_q0circ
            Kq0 = Kq0_circ
        
        return q0, Xt
    
    def update_theta_X(x0:jnp.ndarray,
                theta:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, B_mat, atilde_mat, \
            f_fun, sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, a_fun
        
        U = sp.sim_unif()
        theta_circ = q_sample(theta)
        
        L0circ, M0circ, M0_invcirc, mu0circ, Htcirc, Ftcirc, beta_matcirc, B_matcirc, \
            atilde_matcirc, sigma_fun_thetacirc, logpsi_Xtcirc, rho0_circ, \
                b_fun_thetacirc, Xtcirc = update_mat(theta_circ, Wt)
        
        pi_theta_circ = q_prob(theta_circ)
        pi_theta = q_prob(theta)
        q_theta_circ = q_sample_prob(theta_circ, theta)
        q_theta = q_sample_prob(theta, theta_circ)
        
        A = (rho0_circ*pi_theta_circ*q_theta)/(rho0*pi_theta*q_theta_circ)
        A *= jnp.exp(jnp.sum(logpsi_Xtcirc-logpsi_Xt), axis=0)
        
        if U<A:
            L0 = L0circ
            M0 = M0circ
            M0_inv = M0_invcirc
            mu0 = mu0circ
            Ht = Htcirc
            Ft = Ftcirc
            beta_mat = beta_matcirc
            B_mat = B_matcirc
            atilde_mat = atilde_matcirc
            sigma_fun_theta = sigma_fun_thetacirc
            logpsi_Xt = logpsi_Xtcirc
            rho0 = rho0_circ
            b_fun_theta = b_fun_thetacirc
            Xt = Xtcirc
            theta = theta_circ
            q_theta = q_theta_circ
            
        return theta, Xt
        
    I_obs = len(vT)
    vT = vT.reshape(I_obs, -1)
    nd = vT.shape[-1]
    if theta is not None:
        dim_brown = sigmatilde_fun(time_grid[0], theta).shape[-1]
    else:
        dim_brown = sigmatilde_fun(time_grid[0]).shape[-1]
    if Wt is None:
        Wt = sp.sim_Wt(time_grid, dim=dim_brown, simulations=I_obs)
    if p0 is None:
        p0 = jnp.zeros(nd)
    else:
        p0 = p0.reshape(-1)
    if q0 is None:
        q0 = jnp.mean(vT, axis=0)
    else:
        q0 = q0.reshape(-1)

    x0 = jnp.hstack((q0, p0))  
    
    sqrt_eta = jnp.sqrt(1-eta**2)
    sqrt_deltap = jnp.sqrt(deltap)
    deltap2 = deltap/2
    sqrt_deltaq = jnp.sqrt(deltaq)
    deltaq2 = deltaq/2
    I = jnp.eye(nd)
    Kq0 = kernel_matrix(q0)
    deltapI = deltap*I
    diff_grid = jnp.diff(time_grid, axis=0)

    muT = jnp.zeros(nd)
    L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, B_mat, atilde_mat, \
            sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, Xt \
                = update_mat(theta, Wt)
                
    grad_pL = grad(compute_pLtheta, argnums=0)
    grad_qL = grad(compute_qLtheta, argnums=0)
    L_p0 = grad_pL(p0, Wt)
    L_q0 = grad_qL(q0, Wt)
    pi_x0 = pi_prob(q0,p0)

    if save_step==0:
        if theta is None:
            if update_p0:
                for i in range(max_iter):
                    print("Computing iteration: ", i+1)
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    p0, Xt = update_p0_X(q0, p0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                return Wt, Xt
            else:
                for i in range(max_iter):
                    print("Computing iteration: ", i+1)
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                return Wt, Xt
                
        else:
            theta_list = []
            theta_list.append(theta)
            if update_p0:
                for i in range(max_iter):
                    print("Computing iteration: ", i+1)
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    p0, Xt = update_p0_X(q0, p0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    theta, Xt = update_theta_X(x0, theta, Xt, Wt)
                    theta_list.append(theta)
                return theta_list, Wt, Xt
            else:
                for i in range(max_iter):
                    print("Computing iteration: ", i+1)
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    theta, Xt = update_theta_X(x0, theta, Xt, Wt)
                    theta_list.append(theta)
                return theta_list, Wt, Xt
    else:
        if theta is None:
            if update_p0:
                for i in range(max_iter):
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    p0, Xt = update_p0_X(q0, p0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    
                    if (i+1) % save_step==0:
                        print("Computing iteration: ", i+1)
                        jnp.savez(save_path+'_iter_'+str(i+1), Wt=Wt, Xt=Xt)
                return Wt, Xt
            else:
                for i in range(max_iter):
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    
                    if (i+1) % save_step==0:
                        print("Computing iteration: ", i+1)
                        jnp.savez(save_path+'_iter_'+str(i+1), Wt=Wt, Xt=Xt)
                return Wt, Xt
                
        else:
            theta_list = []
            theta_list.append(theta)
            if update_p0:
                for i in range(max_iter):
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    p0, Xt = update_p0_X(q0, p0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    theta, Xt = update_theta_X(x0, theta, Xt, Wt)
                    theta_list.append(theta)
                    
                    if (i+1) % save_step==0:
                        print("Computing iteration: ", i+1)
                        jnp.savez(save_path+'_iter_'+str(i+1), Wt=Wt, Xt=Xt, 
                                  theta=jnp.vstack(theta_list).squeeze())
                    
                return theta_list, Wt, Xt
            else:
                for i in range(max_iter):
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    theta, Xt = update_theta_X(x0, theta, Xt, Wt)
                    theta_list.append(theta)
                    
                    if (i+1) % save_step==0:
                        print("Computing iteration: ", i+1)
                        jnp.savez(save_path+'_iter_'+str(i+1), Wt=Wt, Xt=Xt, 
                                  theta=jnp.vstack(theta_list).squeeze())
                    
                return jnp.vstack(theta_list).squeeze(), Wt, Xt
    
    
def landmark_template_qT(vT:jnp.ndarray,
                  SigmaT:jnp.ndarray,
                  LT:jnp.ndarray,
                  b_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigma_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  beta_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  B_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigmatilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  k:Callable[[jnp.ndarray], jnp.ndarray],
                  pi_prob:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  time_grid:jnp.ndarray = jnp.linspace(0,1,100),
                  n = 3,
                  d = 1,
                  max_iter:int = 100,
                  eta:float=0.98,
                  deltaq:float=0.01,
                  deltap:float=0.01,
                  theta = None,
                  q_sample = None,
                  q_sample_prob = None,
                  q_prob = None,
                  backward_method = 'odeint',
                  Wt = None,
                  q0 = None,
                  p0 = None,
                  update_p0 = False,
                  save_step = 0,
                  save_path = ''
                  )->Tuple[jnp.ndarray, jnp.ndarray]:
    
    global L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, B_mat, atilde_mat, \
            sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, L_p0, KL_q0, pi_x0, Kq0
            
    def sim_gp(x0, b_fun, sigma_fun, Ft, Ht, Wt):
        
        def update_xi(xi, ite):
            
            t, dt, dWt, F, H = ite
            sigma = sigma_fun(t,xi)
            a = sigma.dot(sigma.T)
            r = F-H.dot(xi)
            x = xi+(b_fun(t,xi)+a.dot(r))*dt+sigma.dot(dWt)
            
            return x, x
        
        dW = jnp.diff(Wt, axis=0)
        _, x = lax.scan(update_xi, init=x0, xs=(time_grid[:-1], diff_grid, dW, 
                                                Ft[:-1], Ht[:-1]))
        
        return jnp.concatenate((x0.reshape(1,-1), x), axis=0)
        
    def logpsi(b_fun, sigma_fun, beta_mat, B_mat, Ft, Ht, atilde_mat, Xt):
        
        b_mat = vmap(b_fun)(time_grid, Xt)
        btilde_mat = beta_mat + jnp.einsum('ijk,ik->ij', B_mat, Xt)
        sigma_mat = vmap(sigma_fun)(time_grid, Xt)
        a_mat = jnp.matmul(sigma_mat, sigma_mat.transpose(0,2,1))
        r_mat = Ft-jnp.einsum('ijk,ik->ij', Ht, Xt)
        
        drift_term = jnp.einsum('ij,ij->i', b_mat-btilde_mat, r_mat)
        Hterm1 = Ht-jnp.einsum('ij,ik->ijk', r_mat,r_mat)
        Hterm2 = a_mat-atilde_mat

        Gx = drift_term-1/2*jnp.trace(jnp.matmul(Hterm1,Hterm2),axis1=1, axis2=2)
        
        val, _ = inter.integrator(Gx, time_grid, method='trapez')
        
        return val
    
    def kernel_matrix(q0):
        
        def compute_row(qi):
            
            if theta is None:
                return vmap(lambda q: k(q,qi))(q0)
            else:
                return vmap(lambda q: k(q,qi,theta))(q0)
        
        q0 = q0.reshape(n,d)
        
        return vmap(compute_row)(q0)

    def update_mat(theta, Wt):
        
        if theta is not None:
            b_fun_theta = lambda t,x: b_fun(t,x,theta)
            sigma_fun_theta = lambda t,x: sigma_fun(t,x,theta)
            beta_fun_theta = lambda t, qT: beta_fun(t,qT, theta)
            B_fun_theta = lambda t, qT: B_fun(t, qT, theta)
            sigmatilde_fun_theta = lambda t, qT: sigmatilde_fun(t, qT, theta)
        else:
            b_fun_theta = b_fun
            sigma_fun_theta = sigma_fun
            beta_fun_theta = beta_fun
            B_fun_theta = B_fun
            sigmatilde_fun_theta = sigmatilde_fun
            
        atilde_fun = lambda t, qT : jnp.dot(sigmatilde_fun_theta(t, qT), \
                                            sigmatilde_fun_theta(t, qT).T)
        beta_mat = vmap(lambda qT: vmap(lambda t: beta_fun_theta(t,qT))(time_grid))(vT)
        B_mat = vmap(lambda qT: vmap(lambda t: B_fun_theta(t,qT))(time_grid))(vT)
        atilde_mat = vmap(lambda qT: vmap(lambda t: atilde_fun(t,qT))(time_grid))(vT)
        
        
        L0 = []
        M0 = []
        mu0 = []
        Ft = []
        Ht = []
        #for i in range(I_obs):
        #    l0, m0, mu, F, H \
        #    = bf.lmmu_step(lambda t: beta_fun_theta(t,vT[i]), 
        #                   lambda t: B_fun_theta(t,vT[i]), 
        #                   lambda t: atilde_fun(t,vT[i]), 
        #                   LT, SigmaT, muT, vT[i], time_grid,
        #                   method=backward_method)
        #    L0.append(l0)
        #    M0.append(m0)
        #    mu0.append(mu)
        #    Ft.append(F)
        #    Ht.append(H)
        #L0 =jnp.stack(L0)
        #M0 =jnp.stack(M0)
        #mu0 =jnp.stack(mu0)
        #Ft =jnp.stack(Ft)
        #Ht =jnp.stack(Ht)
        
        L0, M0, mu0, Ft, Ht = vmap(lambda v: \
            bf.lmmu_step(lambda t: beta_fun_theta(t,v), 
                                  lambda t: B_fun_theta(t,v), 
                                  lambda t: atilde_fun(t,v),
                                  LT, SigmaT, muT, v, time_grid,
                                  method='euler'))(vT)  
        
        M0_inv = jnp.linalg.inv(M0)
        
        Xt = vmap(lambda F,H,w: sim_gp(x0, b_fun_theta, sigma_fun_theta, F, H, w))(Ft,Ht,Wt)
        logpsi_Xt = vmap(lambda beta, B, F, H, a, x: \
                         logpsi(b_fun_theta, sigma_fun_theta, beta, 
                                B, F, H, a, x))(beta_mat, B_mat, Ft, Ht, atilde_mat, Xt)

        rho0 = sp.mnormal_pdf(vT[0], mu0[0]+jnp.einsum('jk,k->j', L0[0], x0), M0[0]) #Maybe vmap
        
        return L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, \
            B_mat, atilde_mat, sigma_fun_theta, logpsi_Xt, \
                rho0, b_fun_theta, Xt
    
    def update_W_X(x0:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:

        global logpsi_Xt
                
        U = sp.sim_unif(dim=I_obs)
        Zt = sp.sim_Wt(time_grid, dim=dim_brown, simulations=I_obs)
        
        Wt_circ = eta*Wt+sqrt_eta*Zt
        Xt_circ = vmap(lambda F,H,w: sim_gp(x0, b_fun_theta, sigma_fun_theta, F, H, w))(Ft,Ht,Wt_circ)
        logpsi_Xtcirc = vmap(lambda beta, B, F, H, a, x: \
                         logpsi(b_fun_theta, sigma_fun_theta, beta, 
                                B, F, H, a, x))(beta_mat, B_mat, Ft, Ht, atilde_mat, Xt_circ)
        
        A = jnp.exp(logpsi_Xtcirc-logpsi_Xt)
        
        bool_val = U<A
        
        Xt = Xt.at[bool_val].set(Xt_circ[bool_val])
        Wt = Wt.at[bool_val].set(Wt_circ[bool_val])
        logpsi_Xt = logpsi_Xt.at[bool_val].set(logpsi_Xtcirc[bool_val])
            
        return Wt, Xt
    
    def compute_pLtheta(p0:jnp.ndarray, Wt:jnp.ndarray)->jnp.ndarray:
            
        x0 = jnp.hstack((q0,p0))
        
        Xt = vmap(lambda F,H,w: sim_gp(x0, b_fun_theta, sigma_fun_theta, F, H, w))(Ft,Ht,Wt)
        logpsi_Xt = vmap(lambda beta, B, F, H, a, x: \
                         logpsi(b_fun_theta, sigma_fun_theta, beta, 
                                B, F, H, a, x))(beta_mat, B_mat, Ft, Ht, atilde_mat, Xt)
        
        x_diff = vT[0]-(mu0[0]+jnp.einsum('jk,k->j', L0[0], x0)) #Maybe vmap
        num = -1/2*(x_diff).T.dot(M0_inv[0]).dot(x_diff)
                
        return jnp.sum(logpsi_Xt) + num
    
    def compute_qLtheta(q0:jnp.ndarray, Wt:jnp.ndarray)->jnp.ndarray:
        
        x0 = jnp.hstack((q0,p0))
        
        Xt = vmap(lambda F,H,w: sim_gp(x0, b_fun_theta, sigma_fun_theta, F, H, w))(Ft,Ht,Wt)
        logpsi_Xt = vmap(lambda beta, B, F, H, a, x: \
                         logpsi(b_fun_theta, sigma_fun_theta, beta, 
                                B, F, H, a, x))(beta_mat, B_mat, Ft, Ht, atilde_mat, Xt)
        
        x_diff = vT[0]-(mu0[0]+jnp.einsum('jk,k->j', L0[0], x0)) #Maybe vmap
        num = -1/2*(x_diff).T.dot(M0_inv[0]).dot(x_diff)
        
        return jnp.sum(logpsi_Xt) + num
    
    def update_p0_X(q0,
                    p0:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global logpsi_Xt, rho0, pi_x0, L_p0
        
        U = sp.sim_unif()
        Z = sp.sim_multinormal(mu=muT,cov=I)
        
        p0_circ = p0+deltap2*L_p0+sqrt_deltap*Z
        x0_circ = jnp.hstack((q0, p0_circ))

        Xt_circ = vmap(lambda F,H,w: sim_gp(x0_circ, b_fun_theta, 
                                            sigma_fun_theta, F, H, w))(Ft,Ht,Wt)
        logpsi_Xtcirc = vmap(lambda beta, B, F, H, a, x: \
                         logpsi(b_fun_theta, sigma_fun_theta, beta, 
                                B, F, H, a, x))(beta_mat, B_mat, Ft, Ht, atilde_mat, Xt_circ)
        
        L_p0circ = grad_pL(p0_circ, Wt)
        if jnp.linalg.norm(deltap2*L_p0circ)>10.0:
            L_p0circ = 0.01*jnp.sign(L_p0circ)/deltap2
        
        rho0_circ = sp.mnormal_pdf(vT[0],mu0[0]+jnp.einsum('jk,k->j', L0[0], x0_circ), M0[0])#Maybe vmap
        pi_x0circ = pi_prob(q0, p0_circ)
        
        norm_p0 = sp.mnormal_pdf(p0, p0_circ+deltap2*L_p0circ, deltapI)
        norm_p0_circ = sp.mnormal_pdf(p0_circ, p0+deltap2*L_p0, deltapI)

        A = ((rho0_circ*pi_x0circ*norm_p0)/(rho0*pi_x0*norm_p0_circ))
        A *= jnp.exp(jnp.sum(logpsi_Xtcirc-logpsi_Xt, axis=0))
                                
        if U<A:
            Xt = Xt_circ
            p0 = p0_circ
            logpsi_Xt = logpsi_Xtcirc
            rho0 = rho0_circ
            pi_x0 = pi_x0circ
            L_p0 = L_p0circ
        
        return p0, Xt
    
    def update_q0_X(q0:jnp.ndarray,
                    p0,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global logpsi_Xt, rho0, pi_x0, KL_q0, Kq0
        
        U = sp.sim_unif()
        Z = sp.sim_multinormal(mu=jnp.zeros(n),cov=Kq0, dim=d).T.reshape(-1)

        q0_circ = q0+deltaq2*KL_q0+sqrt_deltaq*Z
        x0_circ = jnp.hstack((q0_circ, p0))
        
        Xt_circ = vmap(lambda F,H,w: sim_gp(x0_circ, b_fun_theta, 
                                            sigma_fun_theta, F, H, w))(Ft,Ht,Wt)
        logpsi_Xtcirc = vmap(lambda beta, B, F, H, a, x: \
                         logpsi(b_fun_theta, sigma_fun_theta, beta, 
                                B, F, H, a, x))(beta_mat, B_mat, Ft, Ht, atilde_mat, Xt_circ)
        
        Kq0_circ = kernel_matrix(q0_circ)
        KL_q0circ = Kq0_circ.dot(grad_qL(q0_circ, Wt).reshape(n,d)).reshape(-1)
        if jnp.linalg.norm(deltaq2*KL_q0circ)>10.0:
            KL_q0circ = 0.01*jnp.sign(KL_q0circ)/deltaq2
        
        rho0_circ = sp.mnormal_pdf(vT[0],mu0[0]+jnp.einsum('jk,k->j', L0[0], x0_circ), M0[0]) #Maybe vmap
        pi_x0circ = pi_prob(q0_circ, p0)
        Kq0_circ = kernel_matrix(q0_circ)
        
        norm_q0 = jnp.prod(sp.mnormal_pdf(q0.reshape(n,d).T, (q0_circ+deltaq2*KL_q0circ).reshape(n,d).T, 
                                 deltaq*Kq0_circ))
        norm_q0_circ = jnp.prod(sp.mnormal_pdf(q0_circ.reshape(n,d).T, (q0+deltaq2*KL_q0).reshape(n,d).T, 
                                      deltaq*Kq0))
    

        A = ((rho0_circ*pi_x0circ*norm_q0)/(rho0*pi_x0*norm_q0_circ))
        A *= jnp.exp(jnp.sum(logpsi_Xtcirc-logpsi_Xt, axis=0))
                                
        if U<A:
            Xt = Xt_circ
            q0 = q0_circ
            logpsi_Xt = logpsi_Xtcirc
            rho0 = rho0_circ
            pi_x0 = pi_x0circ
            KL_q0 = KL_q0circ
            Kq0 = Kq0_circ
        
        return q0, Xt
    
    def update_theta_X(x0:jnp.ndarray,
                theta:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, B_mat, atilde_mat, \
            f_fun, sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, a_fun, q_theta
        
        U = sp.sim_unif()
        theta_circ = q_sample(theta)
        
        L0circ, M0circ, M0_invcirc, mu0circ, Htcirc, Ftcirc, beta_matcirc, B_matcirc, \
            atilde_matcirc, sigma_fun_thetacirc, logpsi_Xtcirc, rho0_circ, \
                b_fun_thetacirc, Xtcirc = update_mat(theta_circ, Wt)
        
        pi_theta_circ = q_prob(theta_circ)
        pi_theta = q_prob(theta)
        q_theta_circ = q_sample_prob(theta_circ, theta)
        q_theta = q_sample_prob(theta, theta_circ)
        
        A = (rho0_circ*pi_theta_circ*q_theta)/(rho0*pi_theta*q_theta_circ)
        A *= jnp.exp(jnp.sum(logpsi_Xtcirc-logpsi_Xt, axis=0))
        
        if U<A:
            L0 = L0circ
            M0 = M0circ
            M0_inv = M0_invcirc
            mu0 = mu0circ
            Ht = Htcirc
            Ft = Ftcirc
            beta_mat = beta_matcirc
            B_mat = B_matcirc
            atilde_mat = atilde_matcirc
            sigma_fun_theta = sigma_fun_thetacirc
            logpsi_Xt = logpsi_Xtcirc
            rho0 = rho0_circ
            b_fun_theta = b_fun_thetacirc
            Xt = Xtcirc
            theta = theta_circ
            q_theta = q_theta_circ
            
        return theta, Xt
        
    I_obs = len(vT)
    vT = vT.reshape(I_obs, -1)
    nd = vT.shape[-1]
    if theta is not None:
        dim_brown = sigmatilde_fun(time_grid[0], vT[0], theta).shape[-1]
    else:
        dim_brown = sigmatilde_fun(time_grid[0], vT[0]).shape[-1]
    if Wt is None:
        Wt = sp.sim_Wt(time_grid, dim=dim_brown, simulations=I_obs)
    if p0 is None:
        p0 = jnp.ones(nd)*0.0
    else:
        p0 = p0.reshape(-1)
    if q0 is None:
        q0 = jnp.mean(vT, axis=0)
    else:
        q0 = q0.reshape(-1)

    x0 = jnp.hstack((q0, p0))  
    
    sqrt_eta = jnp.sqrt(1-eta**2)
    sqrt_deltap = jnp.sqrt(deltap)
    deltap2 = deltap/2
    sqrt_deltaq = jnp.sqrt(deltaq)
    deltaq2 = deltaq/2
    I = jnp.eye(nd)
    Kq0 = kernel_matrix(q0)
    deltapI = deltap*I
    diff_grid = jnp.diff(time_grid, axis=0)

    muT = jnp.zeros(nd)
    L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, B_mat, atilde_mat, \
            sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, Xt \
                = update_mat(theta, Wt)
                
    grad_pL = grad(compute_pLtheta, argnums=0)
    grad_qL = grad(compute_qLtheta, argnums=0)
    L_p0 = grad_pL(p0, Wt)
    if jnp.linalg.norm(L_p0)>1.0:
        L_p0 = 0.01*jnp.sign(L_p0)
    
    KL_q0 = Kq0.dot(grad_qL(q0, Wt).reshape(n,d)).reshape(-1)
    if jnp.linalg.norm(KL_q0)>1.0:
        KL_q0 = 0.01*jnp.sign(KL_q0)
    
    pi_x0 = pi_prob(q0,p0)

    if save_step==0:
        if theta is None:
            if update_p0:
                for i in range(max_iter):
                    print("Computing iteration: ", i+1)
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    p0, Xt = update_p0_X(q0, p0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                return Wt, Xt
            else:
                for i in range(max_iter):
                    print("Computing iteration: ", i+1)
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                return Wt, Xt
                
        else:
            theta_list = []
            theta_list.append(theta)
            if update_p0:
                for i in range(max_iter):
                    print("Computing iteration: ", i+1)
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    p0, Xt = update_p0_X(q0, p0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    theta, Xt = update_theta_X(x0, theta, Xt, Wt)
                    theta_list.append(theta)
                return theta_list, Wt, Xt
            else:
                for i in range(max_iter):
                    print("Computing iteration: ", i+1)
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    theta, Xt = update_theta_X(x0, theta, Xt, Wt)
                    theta_list.append(theta)
                return theta_list, Wt, Xt
    else:
        if theta is None:
            if update_p0:
                for i in range(max_iter):
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    p0, Xt = update_p0_X(q0, p0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    
                    if (i+1) % save_step==0:
                        print("Computing iteration: ", i+1)
                        jnp.savez(save_path+'_iter_'+str(i+1), Wt=Wt, Xt=Xt)
                return Wt, Xt
            else:
                for i in range(max_iter):
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    
                    if (i+1) % save_step==0:
                        print("Computing iteration: ", i+1)
                        jnp.savez(save_path+'_iter_'+str(i+1), Wt=Wt, Xt=Xt)
                return Wt, Xt
                
        else:
            theta_list = []
            theta_list.append(theta)
            if update_p0:
                for i in range(max_iter):
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    p0, Xt = update_p0_X(q0, p0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    theta, Xt = update_theta_X(x0, theta, Xt, Wt)
                    theta_list.append(theta)
                    
                    if (i+1) % save_step==0:
                        print("Computing iteration: ", i+1)
                        jnp.savez(save_path+'_iter_'+str(i+1), Wt=Wt, Xt=Xt, 
                                  theta=jnp.vstack(theta_list).squeeze())
                    
                return theta_list, Wt, Xt
            else:
                for i in range(max_iter):
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    theta, Xt = update_theta_X(x0, theta, Xt, Wt)
                    theta_list.append(theta)
                    
                    if (i+1) % save_step==0:
                        print("Computing iteration: ", i+1)
                        jnp.savez(save_path+'_iter_'+str(i+1), Wt=Wt, Xt=Xt, 
                                  theta=jnp.vstack(theta_list).squeeze())
                    
                return jnp.vstack(theta_list).squeeze(), Wt, Xt
    