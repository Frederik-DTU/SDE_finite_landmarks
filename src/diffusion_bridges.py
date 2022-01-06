#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:55:05 2021

@author: root
"""
#%% Modules used

import jax.numpy as jnp
from jax import vmap, grad

#Typing declaration of variables
from typing import Callable, List, Tuple

#Own modules
import integration_ode as inter
import backward_filtering as bf
import sim_sp as sp

#%% Functions
    
def logpsi(b, btilde, rtilde, a, atilde, Htilde, grid, Xt=None,
                  method='trapez'):
    
    if callable(b):
        b = vmap(b)(grid, Xt)
    if callable(btilde):
        btilde = vmap(btilde)(grid, Xt)
    if callable(rtilde):
        rtilde = vmap(rtilde)(grid, Xt)
    if callable(a):
        a = vmap(a)(grid, Xt)
    if callable(atilde):
        atilde = vmap(atilde)(grid)
    if callable(Htilde):
        Htilde = vmap(Htilde)(grid)
    
    drift_term = jnp.einsum('ij,ij->i', b-btilde, rtilde)
    
    H_term = Htilde-jnp.einsum('ij,ik->ijk', rtilde, rtilde)
    term2 = jnp.matmul(a-atilde, H_term)
    
    Gx = drift_term-1/2*jnp.trace(term2,axis1=1, axis2=2)
    val, _ = inter.integrator(Gx, grid, method=method)
        
    return val

def landmark_segment(q0:jnp.ndarray,
                  p0:jnp.ndarray,
                  vT:jnp.ndarray,
                  SigmaT:jnp.ndarray,
                  LT:jnp.ndarray,
                  b_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigma_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  betatilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  Btilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigmatilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  pi_prob:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  time_grid:jnp.ndarray = jnp.linspace(0,1,100),
                  max_iter:int = 100,
                  eta:float=0.98,
                  delta:float=0.01,
                  theta = None,
                  q_sample = None,
                  q_prob = None,
                  backward_method = 'odeint',
                  Wt = None,
                  save_step = 0,
                  save_path = ''
                  )->Tuple[jnp.ndarray, jnp.ndarray]:
    
    global Lt0, Mt0, Mt0_inv, mut0, Ht, Ft, betatilde_mat, Btilde_mat, atilde_mat, \
            f_fun, sigma_fun_theta, psi_Xt, rho_x0, b_fun_theta, a_fun, pi_x0
    
    def rtilde_fun(Ft, Ht, xt):
        
        return Ft-jnp.einsum('jk,k->j', Ht, xt)
    
    def update_W_X(x0:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:

        global psi_Xt
                
        U = sp.sim_unif()
        
        Zt = sp.sim_Wt(time_grid, dim_brown)
        Wt_circ = eta*Wt+sqrt_eta*Zt
        Xt_circ = sp.sim_sde_euler(x0, f_fun, sigma_fun_theta, Wt_circ, time_grid)

        b_mat = vmap(b_fun_theta)(time_grid, Xt_circ)
        a_mat = vmap(a_fun)(time_grid, Xt_circ)
        btilde_mat = betatilde_mat+jnp.einsum('ijk, ik->ij', Btilde_mat, Xt_circ)
        rtilde_mat = Ft-jnp.einsum('ijk,ik->ij', Ht, Xt_circ)
        
        psi_Xtcirc = logpsi(b_mat, btilde_mat, rtilde_mat, a_mat, atilde_mat, Ht, time_grid,
                  method='trapez')
        
        A = jnp.exp(psi_Xtcirc-psi_Xt)
        
        if U<A:
            #print("\t-Update (W,X)|(x0,theta,vT) A={:.2}: \t\tAccepted".format(A))
            Xt = Xt_circ
            Wt = Wt_circ
            psi_Xt = psi_Xtcirc
        #else:
            #print("\t-Update (W,X)|(x0,theta,vT) A={:.2}: \t\tRejected".format(A))
            
        return Wt, Xt
    
    def update_p0_X(p0:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global psi_Xt, rho_x0, pi_x0
        
        def compute_Ltheta(p0:jnp.ndarray, Wt:jnp.ndarray)->jnp.ndarray:
            
            x0 = jnp.hstack((q0.reshape(-1),p0.reshape(-1)))
            Xt = sp.sim_sde_euler(x0, f_fun, sigma_fun_theta, Wt, time_grid)
            
            b_mat = vmap(b_fun_theta)(time_grid, Xt)
            a_mat = vmap(a_fun)(time_grid, Xt)
            btilde_mat = betatilde_mat+jnp.einsum('ijk, ik->ij', Btilde_mat, Xt)
            rtilde_mat = Ft-jnp.einsum('ijk,ik->ij', Ht, Xt) #could use vmap
            logpsi_Xt = logpsi(b_mat, btilde_mat, rtilde_mat, a_mat, atilde_mat, Ht, time_grid,
                      method='trapez')
            
            mean = mut0+jnp.einsum('jk,k->j', Lt0, x0)
            x_diff = vT-mean
            
            num = -1/2*(x_diff).T.dot(Mt0_inv).dot(x_diff)
                    
            return logpsi_Xt + num
        
        U = sp.sim_unif()

        Z = sp.sim_multinormal(mu=jnp.zeros(dn),cov=jnp.eye(dn)).reshape(p0.shape)
                     
        grad_L = grad(compute_Ltheta, argnums=0)
        L_p0 = grad_L(p0, Wt)
        
        p0_circ = p0+delta/2*L_p0+sqrt_delta*Z
        x0_circ = jnp.hstack((q0.reshape(-1), p0_circ.reshape(-1)))

        Xt_circ = sp.sim_sde_euler(x0_circ, f_fun, sigma_fun_theta, Wt, time_grid)
        
        L_p0_circ = grad_L(p0_circ, Wt)

        b_mat = vmap(b_fun_theta)(time_grid, Xt_circ)
        a_mat = vmap(a_fun)(time_grid, Xt_circ)
        btilde_mat = betatilde_mat+jnp.einsum('ijk, ik->ij', Btilde_mat, Xt_circ)
        rtilde_mat = Ft-jnp.einsum('ijk,ik->ij', Ht, Xt) #could use vmap
        psi_Xtcirc = logpsi(b_mat, btilde_mat, rtilde_mat, a_mat, atilde_mat, Ht, time_grid,
                  method='trapez')
        
        rho_x0circ = sp.mnormal_pdf(vT,mut0+jnp.einsum('jk,k->j', Lt0, x0_circ), Mt0)

        pi_x0circ = pi_prob(q0, p0_circ)
        norm_p0 = sp.mnormal_pdf(p0.reshape(-1), 
                                 (p0_circ+delta*L_p0_circ/2).reshape(-1),
                                 delta*jnp.eye(dn))
        norm_p0_circ = sp.mnormal_pdf(p0_circ.reshape(-1), 
                                   (p0+delta*L_p0/2).reshape(-1), 
                                   delta*jnp.eye(dn))

        A = ((rho_x0circ*pi_x0circ*norm_p0)/(rho_x0*pi_x0*norm_p0_circ))
        A *= jnp.exp(psi_Xtcirc-psi_Xt)
        
        if jnp.isnan(A):
            A = 1.0
                                
        if U<A:
            #print("\t-Update (p0,X)|(q0,theta,W,vT) A={:.2}: \t\tAccepted".format(A))
            Xt = Xt_circ
            p0 = p0_circ
            psi_Xt = psi_Xtcirc
            rho_x0 = rho_x0circ
            pi_x0 = pi_x0circ
        #else:
            #print("\t-Update (p0,X)|(q0,theta,W,vT) A={:.2}: \t\tRejected".format(A))
        
        return p0, Xt
    
    def update_theta_X(x0:jnp.ndarray,
                theta:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global Lt0, Mt0, Mt0_inv, mut0, Ht, Ft, betatilde_mat, Btilde_mat, atilde_mat, \
            f_fun, sigma_fun_theta, psi_Xt, rho_x0, b_fun_theta, a_fun
        
        U = sp.sim_unif()
        theta_circ = q_sample(theta)
        
        Lt0circ, Mt0circ, mut0circ, Htcirc, Ftcirc, betatilde_matcirc, Btilde_matcirc, \
            atilde_matcirc, f_funcirc, sigma_fun_thetacirc, psi_Xtcirc, rho_x0circ, \
                b_fun_thetacirc, a_funcirc, Xtcirc = update_mat(theta_circ, Wt)
        
        q_theta = q_prob(theta, theta_circ)
        q_theta_circ = q_prob(theta_circ, theta)
        
        A = (rho_x0circ*q_theta)/(rho_x0*q_theta_circ)
        A *= jnp.exp(psi_Xtcirc-psi_Xt)
        
        if U<A:
            #print("\t-Update (theta,X)|(x0,W,vT) A={:.2}: \t\tAccepted".format(A))
            Lt0 = Lt0circ
            Mt0 = Mt0circ
            Mt0_inv = jnp.linalg.norm(Mt0)
            mut0 = mut0circ
            Ht = Htcirc
            Ft = Ftcirc
            betatilde_mat = betatilde_matcirc
            Btilde_mat = Btilde_matcirc
            atilde_mat = atilde_matcirc
            f_fun = f_funcirc
            sigma_fun_theta = sigma_fun_thetacirc
            psi_Xt = psi_Xtcirc
            rho_x0 = rho_x0circ
            b_fun_theta = b_fun_thetacirc
            a_fun = a_funcirc
            Xt = Xtcirc
            theta = theta_circ
        #else:
            #print("\t-Update (theta,X)|(x0,W,vT) A={:.2}: \t\tRejected".format(A))
            
        return theta, Xt
    
    def update_mat(theta, Wt):
        
        if theta is not None:
            b_fun_thetacirc = lambda t,x: b_fun(t,x,theta)
            sigma_fun_thetacirc = lambda t,x: sigma_fun(t,x,theta)
            betatilde_fun_thetacirc = lambda t: betatilde_fun(t,theta)
            Btilde_fun_thetacirc = lambda t: Btilde_fun(t, theta)
            sigmatilde_fun_thetacirc = lambda t: sigmatilde_fun(t, theta)
        else:
            b_fun_thetacirc = b_fun
            sigma_fun_thetacirc = sigma_fun
            betatilde_fun_thetacirc = betatilde_fun
            Btilde_fun_thetacirc = Btilde_fun
            sigmatilde_fun_thetacirc = sigmatilde_fun
    
        a_funcirc = lambda t,x : jnp.dot(sigma_fun_thetacirc(t,x), \
                                         sigma_fun_thetacirc(t,x).T)
        atilde_funcirc = lambda t : jnp.dot(sigmatilde_fun_thetacirc(t), \
                                            sigmatilde_fun_thetacirc(t).T)
        
        Ltcirc0, Mtcirc0, mutcirc0, Ftcirc, Htcirc \
            = bf.lmmu_step(betatilde_fun_thetacirc, Btilde_fun_thetacirc, 
                           atilde_funcirc, LT, SigmaT, muT, vT, time_grid,
                           method=backward_method)
                
        rtildecirc = lambda t,x: \
            rtilde_fun(Ftcirc[jnp.argmin(jnp.abs(time_grid-t))], Htcirc[jnp.argmin(jnp.abs(time_grid-t))], x)
        f_funcirc = lambda t,x: b_fun_thetacirc(t,x)+a_funcirc(t,x).dot(rtildecirc(t,x))

        Xt_circ = sp.sim_sde_euler(x0, f_funcirc, sigma_fun_thetacirc, Wt, time_grid)
        
        betatilde_matcirc = vmap(betatilde_fun_thetacirc)(time_grid)
        Btilde_matcirc = vmap(Btilde_fun_thetacirc)(time_grid)
        atilde_matcirc = vmap(atilde_funcirc)(time_grid)
        btilde_matcirc = betatilde_matcirc+jnp.einsum('ijk, ik->ij', Btilde_matcirc, Xt_circ)
        rtilde_matcirc = Ftcirc-jnp.einsum('ijk,ik->ij', Htcirc, Xt_circ) #could use vmap
        
        psi_Xtcirc = logpsi(b_fun_thetacirc, btilde_matcirc, rtilde_matcirc, 
                            a_funcirc, atilde_matcirc, Htcirc, time_grid,
                            Xt_circ, method='trapez')
        rho_x0circ = sp.mnormal_pdf(vT, mutcirc0+jnp.einsum('jk,k->j', Ltcirc0, x0), Mtcirc0)
        
        return Ltcirc0, Mtcirc0, mutcirc0, Htcirc, Ftcirc, betatilde_matcirc, \
            Btilde_matcirc, atilde_matcirc, f_funcirc, sigma_fun_thetacirc, psi_Xtcirc, \
                rho_x0circ, b_fun_thetacirc, a_funcirc, Xt_circ
        
    if theta is not None:
        sigmatilde_fun_theta = lambda t: sigmatilde_fun(t, theta)
    else:
        sigmatilde_fun_theta = sigmatilde_fun
        
    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
    sqrt_eta = jnp.sqrt(1-eta**2)
    sqrt_delta = jnp.sqrt(delta)
    dn = len(q0.reshape(-1))
    dim_brown = sigmatilde_fun_theta(time_grid[0]).shape[-1]
    muT = jnp.zeros_like(q0.reshape(-1))
    
    if Wt is None:
        Wt = sp.sim_Wt(time_grid, dim_brown)
    Lt0, Mt0, mut0, Ht, Ft, betatilde_mat, Btilde_mat, atilde_mat, \
            f_fun, sigma_fun_theta, psi_Xt, rho_x0, b_fun_theta, a_fun, Xt \
                = update_mat(theta, Wt)
    Mt0_inv = jnp.linalg.norm(Mt0)

    pi_x0 = pi_prob(q0,p0)

    if save_step==0:
        if theta is None:
            for i in range(max_iter):
                print("Computing iteration: ", i+1)
                Wt, Xt = update_W_X(x0, Xt, Wt)
                p0, Xt = update_p0_X(p0, Xt, Wt)
                x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                #print(Xt[-1]) #Only to print iterations throughout 
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
    else:
        if theta is None:
            for i in range(max_iter):
                Wt, Xt = update_W_X(x0, Xt, Wt)
                p0, Xt = update_p0_X(p0, Xt, Wt)
                x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                
                if (i+1) % save_step==0:
                    print("Computing iteration: ", i+1)
                    jnp.savez(save_path+'Wt_Xt_'+str(i+1), Wt=Wt, Xt=Xt)
                
                #print(Xt[-1]) #Only to print iterations throughout 
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
                    jnp.savez(save_path+'Wt_Xt_theta'+str(i+1), Wt=Wt, Xt=Xt, 
                              theta=theta_list)
            
        return theta_list, Wt, Xt
