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
    
    grad_L = grad(compute_Ltheta, argnums=0)
    
    def update_p0_X(p0:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global psi_Xt, rho_x0, pi_x0
        
        U = sp.sim_unif()

        Z = sp.sim_multinormal(mu=jnp.zeros(dn),cov=jnp.eye(dn)).reshape(p0.shape)
                     
        L_p0 = grad_L(p0, Wt)
        
        p0_circ = p0+delta/2*L_p0+sqrt_delta*Z
        x0_circ = jnp.hstack((q0.reshape(-1), p0_circ.reshape(-1)))

        Xt_circ = sp.sim_sde_euler(x0_circ, f_fun, sigma_fun_theta, Wt, time_grid)
        
        L_p0_circ = grad_L(p0_circ, Wt)

        b_mat = vmap(b_fun_theta)(time_grid, Xt_circ)
        a_mat = vmap(a_fun)(time_grid, Xt_circ)
        btilde_mat = betatilde_mat+jnp.einsum('ijk, ik->ij', Btilde_mat, Xt_circ)
        rtilde_mat = Ft-jnp.einsum('ijk,ik->ij', Ht, Xt_circ) #could use vmap
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
                print(i)
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
    
def landmark_template_q0(q0:jnp.ndarray,
                  vT:jnp.ndarray,
                  SigmaT:jnp.ndarray,
                  LT:jnp.ndarray,
                  b_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigma_fun:Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  betatilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  Btilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  sigmatilde_fun:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  pi_prob:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                  k:Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
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
    
    def compute_K(q):
        
        if theta is not None:
            k = lambda x: k(x, theta)
        
        def Kqi(qi)->jnp.ndarray:
            
            K = vmap(k)(qi-q,theta)
            
            return K
            
        return vmap(Kqi)(q)
    
    def update_mat(theta, Wt):
        
        def update_xt(carry, Wi):
            
            Xi = sp.sim_sde_euler(x0, f_funcirc, sigma_fun_thetacirc, Wi, time_grid)
        
            btilde_matcirc = betatilde_matcirc+jnp.einsum('ijk, ik->ij', Btilde_matcirc, Xi)
            rtilde_matcirc = Ftcirc-jnp.einsum('ijk,ik->ij', Htcirc, Xi) #could use vmap
            
            psi_Xtcirc = logpsi(b_fun_thetacirc, btilde_matcirc, rtilde_matcirc, 
                                a_funcirc, atilde_matcirc, Htcirc, time_grid,
                                Xi, method='trapez')
            
            res = (Xi, btilde_matcirc, rtilde_matcirc, psi_Xtcirc)
            
            return res, res
        
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
                           atilde_funcirc, LT, SigmaT, muT, vT[0], time_grid,
                           method=backward_method)
                
        rtildecirc = lambda t,x: \
            rtilde_fun(Ftcirc[jnp.argmin(jnp.abs(time_grid-t))], Htcirc[jnp.argmin(jnp.abs(time_grid-t))], x)
        f_funcirc = lambda t,x: b_fun_thetacirc(t,x)+a_funcirc(t,x).dot(rtildecirc(t,x))
        betatilde_matcirc = vmap(betatilde_fun_thetacirc)(time_grid)
        Btilde_matcirc = vmap(Btilde_fun_thetacirc)(time_grid)
        atilde_matcirc = vmap(atilde_funcirc)(time_grid)
        rho_x0circ = sp.mnormal_pdf(vT, mutcirc0+jnp.einsum('jk,k->j', Ltcirc0, x0), Mtcirc0)
        
        ##### THIS SHOULD BE MODIFIED FOR SPEED
        Xi = sp.sim_sde_euler(x0, f_funcirc, sigma_fun_thetacirc, Wt[0], time_grid)
        btilde_matcirc = betatilde_matcirc+jnp.einsum('ijk, ik->ij', Btilde_matcirc, Xi)
        rtilde_matcirc = Ftcirc-jnp.einsum('ijk,ik->ij', Htcirc, Xi) #could use vmap
        
        psi_Xtcirc = logpsi(b_fun_thetacirc, btilde_matcirc, rtilde_matcirc, 
                            a_funcirc, atilde_matcirc, Htcirc, time_grid,
                            Xi, method='trapez')
        _, y = lax.scan(update_xt, init=(Xi, btilde_matcirc, rtilde_matcirc, psi_Xtcirc), 
                        xs=Wt)
        #### THIS^
        
        Xt_circ, btilde_matcirc, rtilde_matcirc, psy_Xtcirc = y
        
        return Ltcirc0, Mtcirc0, mutcirc0, Htcirc, Ftcirc, betatilde_matcirc, \
            Btilde_matcirc, atilde_matcirc, f_funcirc, sigma_fun_thetacirc, psi_Xtcirc, \
                rho_x0circ, b_fun_thetacirc, a_funcirc, Xt_circ
    
    def update_W_X(x0:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global psi_Xt
        
        def update_path(carry, Wi):
            
            #carry = (Wi_circ, Xi_circ, psi_Xicirc)
            
            Zi = sp.sim_Wt(time_grid, dim_brown)
            Wi_circ = eta*Wi+sqrt_eta*Zi
            Xi_circ = sp.sim_sde_euler(x0, f_fun, sigma_fun_theta, Wi_circ, time_grid)
    
            b_mat = vmap(b_fun_theta)(time_grid, Xi_circ)
            a_mat = vmap(a_fun)(time_grid, Xi_circ)
            btilde_mat = betatilde_mat+jnp.einsum('ijk, ik->ij', Btilde_mat, Xi_circ)
            rtilde_mat = Ft-jnp.einsum('ijk,ik->ij', Ht, Xi_circ)
            
            psi_Xicirc = logpsi(b_mat, btilde_mat, rtilde_mat, a_mat, atilde_mat, Ht, time_grid,
                      method='trapez')
            
            res = (Wi_circ, Xi_circ, psi_Xicirc)
            
            return res, res
                
        U = sp.sim_unif(dim=I)
        init = (Wt[0], Xt[0], psi_Xt[0])
        
        _, y = lax.scan(update_path, init, Wt)
        Wt_circ, Xt_circ, psi_Xtcirc = y
        
        A = jnp.exp(psi_Xtcirc-psi_Xt)
        
        bool_val = U<A
        Xt = Xt.at[bool_val].set(Xt_circ[bool_val])
        Wt = Wt.at[bool_val].set(Wt_circ[bool_val])
        psi_Xt = psi_Xt.at[bool_val].set(psi_Xtcirc[bool_val])
            
        return Wt, Xt
    
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
        A *= jnp.exp(jnp.sum(psi_Xtcirc-psi_Xt, axis=0))
        
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
    
    def update_q0_X(q0:jnp.ndarray,
                Xt:jnp.ndarray,
                Wt:jnp.ndarray)->Tuple[jnp.ndarray, jnp.ndarray]:
        
        global psi_Xt, rho_x0, pi_x0
        
        def compute_Ltheta(q0:jnp.ndarray, Wt:jnp.ndarray)->jnp.ndarray:
            
            def update_xt(Wi):
            
                Xi = sp.sim_sde_euler(x0, f_fun, sigma_fun_theta, Wi, time_grid)
            
                btilde_mat = betatilde_mat+jnp.einsum('ijk, ik->ij', Btilde_mat, Xi)
                rtilde_mat = Ft-jnp.einsum('ijk,ik->ij', Ht, Xi) #could use vmap
                
                psi_Xtcirc = logpsi(b_fun_theta, btilde_mat, rtilde_mat, 
                                    a_fun, atilde_mat, Ht, time_grid,
                                    Xi, method='trapez')
                
                return psi_Xtcirc
            
            x0 = jnp.hstack((q0.reshape(-1),p0.reshape(-1)))
            
            logpsi_Xt = jnp.sum(vmap(update_xt)(Wt), axis=0)
            
            mean = mut0+jnp.einsum('jk,k->j', Lt0, x0)
            x_diff = vT-mean
            
            num = -1/2*(x_diff).T.dot(Mt0_inv).dot(x_diff)
                    
            return logpsi_Xt + num
        
        U = sp.sim_unif()

        Kq = compute_K(q0)
        Z = sp.sim_multinormal(mu=jnp.zeros(dn),cov=Kq).reshape(p0.shape)
                     
        grad_L = grad(compute_Ltheta, argnums=0)
        L_q0 = grad_L(q0, Wt)
        
        q0_circ = p0+delta/2*L_q0+sqrt_delta*Z
        Kq_circ = compute_K(q0_circ)
        x0_circ = jnp.hstack((q0_circ.reshape(-1), p0.reshape(-1)))
        
        #Should be fixed such that psi_xtcirc is also computed
        Xt_circ = vmap(sp.sim_sde_euler)(x0_circ, f_fun, sigma_fun_theta, Wt, time_grid)
        
        L_q0_circ = grad_L(q0_circ, Wt)

        #Should be modified
        b_mat = vmap(b_fun_theta)(time_grid, Xt_circ)
        a_mat = vmap(a_fun)(time_grid, Xt_circ)
        btilde_mat = betatilde_mat+jnp.einsum('ijk, ik->ij', Btilde_mat, Xt_circ)
        rtilde_mat = Ft-jnp.einsum('ijk,ik->ij', Ht, Xt) #could use vmap
        psi_Xtcirc = logpsi(b_mat, btilde_mat, rtilde_mat, a_mat, atilde_mat, Ht, time_grid,
                  method='trapez')
        #^
        
        rho_x0circ = sp.mnormal_pdf(vT,mut0+jnp.einsum('jk,k->j', Lt0, x0_circ), Mt0)

        pi_x0circ = pi_prob(q0_circ, p0)
        norm_p0 = sp.mnormal_pdf(q0.reshape(-1), 
                                 (q0_circ+delta*jnp.matmul(Kq_circ,L_q0_circ)/2).reshape(-1),
                                 delta*Kq_circ)
        norm_p0_circ = sp.mnormal_pdf(q0_circ.reshape(-1), 
                                 (q0+delta*jnp.matmul(Kq,L_q0)/2).reshape(-1),
                                 delta*Kq)

        A = ((rho_x0circ*pi_x0circ*norm_p0)/(rho_x0*pi_x0*norm_p0_circ))
        A *= jnp.exp(jnp.sum(psi_Xtcirc-psi_Xt, axis=0))
        
        if jnp.isnan(A):
            A = 1.0
                                
        if U<A:
            #print("\t-Update (p0,X)|(q0,theta,W,vT) A={:.2}: \t\tAccepted".format(A))
            Xt = Xt_circ
            q0 = q0_circ
            psi_Xt = psi_Xtcirc
            rho_x0 = rho_x0circ
            pi_x0 = pi_x0circ
        #else:
            #print("\t-Update (p0,X)|(q0,theta,W,vT) A={:.2}: \t\tRejected".format(A))
        
        return q0, Xt
    
    p0 = jnp.zeros_like(q0)
    x0 = jnp.hstack((q0.reshape(-1),p0.reshape(-1)))
    I = len(vT)
    dim_brown = sigmatilde_fun(time_grid[0]).shape[-1]
    sqrt_eta = jnp.sqrt(1-eta**2)
    sqrt_delta = jnp.sqrt(delta)
    muT = jnp.zeros_like(q0.reshape(-1))
    dn = len(muT)
    Wt = jnp.zeros((I,len(time_grid), dim_brown))
    for i in range(I):
        Wt = Wt.at[i].set(sp.sim_Wt(grid = time_grid, dim=dim_brown))
    
    
    
    return

#%% The garage

def landmark_segment2(q0:jnp.ndarray,
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
                  q_prob = None,
                  backward_method = 'odeint',
                  Wt = None,
                  p0:jnp.ndarray = None,
                  save_step = 0,
                  save_path = ''
                  )->Tuple[jnp.ndarray, jnp.ndarray]:
    
    global L0, M0, M0_inv, mu0, Ht, Ft, beta_mat, B_mat, atilde_mat, \
            sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, L_p0, pi_x0, q_theta
            
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
                           atilde_fun, LT, SigmaT, muT, vT, time_grid,
                           method=backward_method)
            
        M0_inv = jnp.linalg.inv(M0)
        
        Xt = sim_gp(x0, b_fun_theta, sigma_fun_theta, Ft, Ht, Wt)
        logpsi_Xt = logpsi(b_fun_theta, sigma_fun_theta, beta_mat, B_mat, 
                           Ft, Ht, atilde_mat, Xt)
        rho0 = sp.mnormal_pdf(vT, mu0+jnp.einsum('jk,k->j', L0, x0), M0)
        
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
        
        x_diff = vT-(mu0+jnp.einsum('jk,k->j', L0, x0))
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
        rho0_circ = sp.mnormal_pdf(vT,mu0+jnp.einsum('jk,k->j', L0, x0_circ), M0)
        pi_x0circ = pi_prob(q0, p0_circ)
        
        norm_p0 = sp.mnormal_pdf(p0, p0_circ+delta2*L_p0circ, deltaI)
        norm_p0_circ = sp.mnormal_pdf(p0_circ, p0+delta2*L_p0, deltaI)

        A = ((rho0_circ*pi_x0circ*norm_p0)/(rho0*pi_x0*norm_p0_circ))
        A *= jnp.exp(logpsi_Xtcirc-logpsi_Xt)
        
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
        
        q_theta_circ = q_prob(theta_circ)
        
        A = (rho0_circ*q_theta)/(rho0*q_theta_circ)
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
    if theta is not None:
        q_theta = q_prob(theta)
        
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
                    jnp.savez(save_path+'Wt_Xt_'+str(i+1), Wt=Wt, Xt=Xt)
                
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
            sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, L_p0, L_q0, pi_x0, Kq0, q_theta
            
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
        
        if jnp.isnan(A) or jnp.isinf(A):
            A = 1.0
                                
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
            f_fun, sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, a_fun, q_theta
        
        U = sp.sim_unif()
        theta_circ = q_sample(theta)
        
        L0circ, M0circ, M0_invcirc, mu0circ, Htcirc, Ftcirc, beta_matcirc, B_matcirc, \
            atilde_matcirc, sigma_fun_thetacirc, logpsi_Xtcirc, rho0_circ, \
                b_fun_thetacirc, Xtcirc = update_mat(theta_circ, Wt)
        
        q_theta_circ = q_prob(theta_circ)
        
        A = (rho0_circ*q_theta)/(rho0*q_theta_circ)
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
    if theta is not None:
        q_theta = q_prob(theta)

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
                        jnp.savez(save_path+'Wt_Xt_'+str(i+1), Wt=Wt, Xt=Xt)
                return Wt, Xt
            else:
                for i in range(max_iter):
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    
                    if (i+1) % save_step==0:
                        print("Computing iteration: ", i+1)
                        jnp.savez(save_path+'Wt_Xt_'+str(i+1), Wt=Wt, Xt=Xt)
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
                        jnp.savez(save_path+'Wt_Xt_theta'+str(i+1), Wt=Wt, Xt=Xt, 
                                  theta=theta_list)
                    
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
                        jnp.savez(save_path+'Wt_Xt_theta'+str(i+1), Wt=Wt, Xt=Xt, 
                                  theta=theta_list)
                    
                return theta_list, Wt, Xt
    
    
def landmark_template2(vT:jnp.ndarray,
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
            sigma_fun_theta, logpsi_Xt, rho0, b_fun_theta, L_p0, KL_q0, pi_x0, Kq0, q_theta
            
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
        rho0_circ = sp.mnormal_pdf(vT[0],mu0[0]+jnp.einsum('jk,k->j', L0[0], x0_circ), M0[0])#Maybe vmap
        pi_x0circ = pi_prob(q0, p0_circ)
        
        norm_p0 = sp.mnormal_pdf(p0, p0_circ+deltap2*L_p0circ, deltapI)
        norm_p0_circ = sp.mnormal_pdf(p0_circ, p0+deltap2*L_p0, deltapI)

        A = ((rho0_circ*pi_x0circ*norm_p0)/(rho0*pi_x0*norm_p0_circ))
        A *= jnp.exp(jnp.sum(logpsi_Xtcirc-logpsi_Xt, axis=0))
        
        if jnp.isnan(A) or jnp.isinf(A):
            A = 0.0
                                
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
        Z = sp.sim_multinormal(mu=muT,cov=Kq0)
        
        q0_circ = q0+deltaq2*KL_q0+sqrt_deltaq*Z
        x0_circ = jnp.hstack((q0_circ, p0))
        
        Xt_circ = vmap(lambda F,H,w: sim_gp(x0_circ, b_fun_theta, 
                                            sigma_fun_theta, F, H, w))(Ft,Ht,Wt)
        logpsi_Xtcirc = vmap(lambda beta, B, F, H, a, x: \
                         logpsi(b_fun_theta, sigma_fun_theta, beta, 
                                B, F, H, a, x))(beta_mat, B_mat, Ft, Ht, atilde_mat, Xt_circ)
        
        Kq0_circ = kernel_matrix(q0_circ)
        KL_q0circ = Kq0_circ.dot(grad_qL(q0_circ, Wt))
        rho0_circ = sp.mnormal_pdf(vT[0],mu0[0]+jnp.einsum('jk,k->j', L0[0], x0_circ), M0[0]) #Maybe vmap
        pi_x0circ = pi_prob(q0_circ, p0)
        Kq0_circ = kernel_matrix(q0_circ)
        
        norm_q0 = sp.mnormal_pdf(q0, q0_circ+deltaq2*KL_q0circ, 
                                 deltaq*Kq0_circ)
        norm_q0_circ = sp.mnormal_pdf(q0_circ, q0+deltaq2*KL_q0, 
                                      deltaq*Kq0)

        A = ((rho0_circ*pi_x0circ*norm_q0)/(rho0*pi_x0*norm_q0_circ))
        A *= jnp.exp(jnp.sum(logpsi_Xtcirc-logpsi_Xt, axis=0))

        if jnp.isnan(A) or jnp.isinf(A):
            A = 0.0
                                
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
        
        q_theta_circ = q_prob(theta_circ)
        
        A = (rho0_circ*q_theta)/(rho0*q_theta_circ)
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
    if theta is not None:
        q_theta = q_prob(theta)

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
    KL_q0 = Kq0.dot(grad_qL(q0, Wt))
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
                        jnp.savez(save_path+'Wt_Xt_'+str(i+1), Wt=Wt, Xt=Xt)
                return Wt, Xt
            else:
                for i in range(max_iter):
                    Wt, Xt = update_W_X(x0, Xt, Wt)
                    q0, Xt = update_q0_X(q0, p0, Xt, Wt)
                    x0 = jnp.hstack((q0.reshape(-1), p0.reshape(-1)))
                    
                    if (i+1) % save_step==0:
                        print("Computing iteration: ", i+1)
                        jnp.savez(save_path+'Wt_Xt_'+str(i+1), Wt=Wt, Xt=Xt)
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
                        jnp.savez(save_path+'Wt_Xt_theta'+str(i+1), Wt=Wt, Xt=Xt, 
                                  theta=theta_list)
                    
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
                        jnp.savez(save_path+'Wt_Xt_theta'+str(i+1), Wt=Wt, Xt=Xt, 
                                  theta=theta_list)
                    
                return theta_list, Wt, Xt
    