#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 19:48:25 2021

@author: root
"""

#%% Modules

import jax.numpy as jnp
from jax import vmap, lax

import math

from scipy.integrate import odeint

from typing import List

#Own modules
import integration_ode as inter

#%% Functions

def lmmu_step_template(beta, B_fun, a, LT, MT, muT, vT, time_grid, method='odeint'):
    
    def compute_Ht(Lt, Mt_inv):
        
        val = jnp.einsum('ijn,ink->ijk', Lt.transpose(0,2,1), Mt_inv)
        
        return jnp.einsum('ijn,ink->ijk', val, Lt)
    
    def compute_Ft(Lt, Mt_inv, vt, mut):

        val = jnp.einsum('ijn,ink->ijk', Lt.transpose(0,2,1), Mt_inv)
        
        return jnp.einsum('inj,ij->in', val, vt-mut)
    
    def euler_method():
        
        if callable(beta):
            beta_mat = vmap(beta)(time_grid)[..., jnp.newaxis]
        else:
            beta_mat = beta
        if callable(a):
            a_mat = vmap(a)(time_grid)
        else:
            a_mat = a
        
        grid_reverse = time_grid[::-1]
        
        dL = lambda t,L : -jnp.dot(L,B_fun(t))
        Lt = inter.ode_integrator(LT, dL, grid_reverse, method='euler')[::-1]
        
        dM = -jnp.matmul(jnp.matmul(Lt, a_mat),Lt.transpose(0,2,1))
        dmu = -jnp.matmul(Lt, beta_mat).reshape(len(dM),-1)
    
        _, Mt = inter.integrator(dM[::-1], grid_reverse, method='trapez')
        _, mut = inter.integrator(dmu[::-1], grid_reverse, method='trapez')
        
        Mt = MT+Mt[::-1]
        mut = muT+mut[::-1]
        Mt_inv = jnp.linalg.inv(Mt)
        Ft = vmap(lambda v: compute_Ft(Lt, Mt_inv, v, mut))(vT)
        Ht = compute_Ht(Lt, Mt_inv)
        
        return Lt[0], Mt[0], mut[0], Ft, Ht
    
    def odeint_method():
        
        def backward_fun(y:jnp.ndarray, 
                           t:jnp.ndarray,
                           LT_dim:List[int],
                           LT_dim_flatten:int)->jnp.ndarray:
            
            Lt = y[0:LT_dim_flatten].reshape(LT_dim)
            
            dLt = -jnp.dot(Lt, B_fun(t)) #dL(t)=-L(t)B(t)dt
            dM = -jnp.dot(jnp.dot(Lt, a(t)), Lt.T) #dM(t)=-L(t)a(t)L(t)'dt
            dmu = -jnp.dot(Lt, beta(t))
            
            rhs = jnp.hstack((dLt.reshape(-1), dM.reshape(-1), dmu.reshape(-1)))
                            
            return rhs
        
        grid_reverse = time_grid[::-1]
    
        LT_dim = list(LT.shape)
        LT_dim_flatten = math.prod(LT_dim)
        SigmaT_dim = list(MT.shape)
        SigmaT_dim_flatten = math.prod(SigmaT_dim)
            
        yT = jnp.hstack((LT.reshape(-1), MT.reshape(-1), 
                         muT))
                
        y = odeint(backward_fun, yT, grid_reverse, 
                   args=(LT_dim,LT_dim_flatten))
        
        y = y[::-1] #Reversing the array such that yT=yT and y0=y0
        
        Lt = y[:, 0:LT_dim_flatten]
        Mt = y[:, LT_dim_flatten:(LT_dim_flatten+SigmaT_dim_flatten)]
        mut = y[:,(LT_dim_flatten+SigmaT_dim_flatten):]
                
        Lt = Lt.reshape([-1]+LT_dim)
        Mt = Mt.reshape([-1]+SigmaT_dim)
        Mt_inv = jnp.linalg.inv(Mt)
        
        Ft = vmap(lambda v: compute_Ft(Lt, Mt_inv, v, mut))(vT)
        Ht = compute_Ht(Lt, Mt_inv)
        
        return Lt[0], Mt[0], mut[0], Ft, Ht
    
    if len(LT.shape)==1:
        LT = LT.reshape(1,-1)
    if len(MT.shape)==1:
        MT = MT.reshape(1,1)
    
    if method=='odeint':
        return odeint_method() #Lt, Mt, mut, Ft, Ht
    else:
        return euler_method() #Lt, Mt, mut, Ft, Ht

def lmmu_step(beta, B_fun, a, LT, MT, muT, vt, time_grid, method='odeint'):
    
    def compute_Ht(Lt, Mt_inv):
        
        val = jnp.einsum('ijn,ink->ijk', Lt.transpose(0,2,1), Mt_inv)
        
        return jnp.einsum('ijn,ink->ijk', val, Lt)
    
    def compute_Ft(Lt, Mt_inv, vt, mut):

        val = jnp.einsum('ijn,ink->ijk', Lt.transpose(0,2,1), Mt_inv)
        
        return jnp.einsum('inj,ij->in', val, vt-mut)
    
    def euler_method():
        
        if callable(beta):
            beta_mat = vmap(beta)(time_grid)[..., jnp.newaxis]
        else:
            beta_mat = beta
        if callable(a):
            a_mat = vmap(a)(time_grid)
        else:
            a_mat = a
        
        grid_reverse = time_grid[::-1]
        
        dL = lambda t,L : -jnp.dot(L,B_fun(t))
        Lt = inter.ode_integrator(LT, dL, grid_reverse, method='euler')[::-1]
        
        dM = -jnp.matmul(jnp.matmul(Lt, a_mat),Lt.transpose(0,2,1))
        dmu = -jnp.matmul(Lt, beta_mat).reshape(len(dM),-1)
    
        _, Mt = inter.integrator(dM[::-1], grid_reverse, method='trapez')
        _, mut = inter.integrator(dmu[::-1], grid_reverse, method='trapez')
        
        Mt = MT+Mt[::-1]
        mut = muT+mut[::-1]
        Mt_inv = jnp.linalg.inv(Mt)
        Ft = compute_Ft(Lt, Mt_inv, vt, mut)
        Ht = compute_Ht(Lt, Mt_inv)
        
        return Lt[0], Mt[0], mut[0], Ft, Ht
    
    def odeint_method():
        
        def backward_fun(y:jnp.ndarray, 
                           t:jnp.ndarray,
                           LT_dim:List[int],
                           LT_dim_flatten:int)->jnp.ndarray:
            
            Lt = y[0:LT_dim_flatten].reshape(LT_dim)
            
            dLt = -jnp.dot(Lt, B_fun(t)) #dL(t)=-L(t)B(t)dt
            dM = -jnp.dot(jnp.dot(Lt, a(t)), Lt.T) #dM(t)=-L(t)a(t)L(t)'dt
            dmu = -jnp.dot(Lt, beta(t))
            
            rhs = jnp.hstack((dLt.reshape(-1), dM.reshape(-1), dmu.reshape(-1)))
                            
            return rhs
        
        grid_reverse = time_grid[::-1]
    
        LT_dim = list(LT.shape)
        LT_dim_flatten = math.prod(LT_dim)
        SigmaT_dim = list(MT.shape)
        SigmaT_dim_flatten = math.prod(SigmaT_dim)
        
        yT = jnp.hstack((LT.reshape(-1), MT.reshape(-1), 
                         muT))
                
        y = odeint(backward_fun, yT, grid_reverse, 
                   args=(LT_dim,LT_dim_flatten))
        
        y = y[::-1] #Reversing the array such that yT=yT and y0=y0
        
        Lt = y[:, 0:LT_dim_flatten]
        Mt = y[:, LT_dim_flatten:(LT_dim_flatten+SigmaT_dim_flatten)]
        mut = y[:,(LT_dim_flatten+SigmaT_dim_flatten):]
                
        Lt = Lt.reshape([-1]+LT_dim)
        Mt = Mt.reshape([-1]+SigmaT_dim)
        Mt_inv = jnp.linalg.inv(Mt)

        Ft = compute_Ft(Lt, Mt_inv, vt, mut)
        Ht = compute_Ht(Lt, Mt_inv)
        
        return Lt[0], Mt[0], mut[0], Ft, Ht
    
    if len(LT.shape)==1:
        LT = LT.reshape(1,-1)
    if len(MT.shape)==1:
        MT = MT.reshape(1,1)
    
    if method=='odeint':
        return odeint_method() #Lt, Mt, mut, Ft, Ht
    else:
        return euler_method() #Lt, Mt, mut, Ft, Ht
    
def lmmu(beta_fun, B_fun, a_fun, P_n, L_list, Sigma_list, v_list, t, grid_fun, method='odeint'):
    
    def compute_Ht(Lt, Mt_inv):
        
        val = jnp.einsum('jn,nk->jk', Lt.T, Mt_inv)
        
        return jnp.einsum('jn,nk->jk', val, Lt)
    
    def compute_Ft(Lt, Mt_inv, vt, mut):

        val = jnp.einsum('jn,nk->jk', Lt.T, Mt_inv)
        
        return jnp.einsum('nj,j->n', val, vt-mut)
    
    n_steps = len(L_list)
    n_states = len(P_n)
    
    mi = len(v_list[-1])
    zero_mat = jnp.zeros((mi, n_states))
    LT = jnp.vstack((jnp.eye(n_states), L_list[-1]))
    MT = jnp.block([[Sigma_list[-1], zero_mat],
                     [zero_mat.T, P_n]])
    muT = jnp.hstack((jnp.zeros(mi), jnp.zeros(n_states)))
    vt = jnp.concatenate((v_list[-1], jnp.zeros(n_states)))
    
    
    Lt_list = []
    Mt_list = []
    mut_list = []
    Ft_list = []
    Ht_list = []
    for i in range(n_steps-1, 0, -1):
        
        time_grid = grid_fun(t[i-1], t[i])
        Lt, Mt, mut, Ft, Ht = lmmu_step(beta_fun, B_fun, a_fun, LT, MT, muT, vt, time_grid, method)
        Lt_list.append(Lt)
        Mt_list.append(Mt)
        mut_list.append(mut)
        Ft_list.append(Ft)
        Ht_list.append(Ht)

        mi = len(v_list[i])
        mt = len(vt)
        zero_mat = jnp.zeros((mi, mt))
        LT = jnp.vstack((L_list[i], Lt))
        MT = jnp.block([[Sigma_list[i], zero_mat],
                         [zero_mat.T, Mt]])
        muT = jnp.hstack((jnp.zeros(mi), mut))
        vt = jnp.concatenate((v_list[i], vt))
    
    MT_inv = jnp.linalg.inv(MT)
    Lt_list.append(LT)
    Mt_list.append(MT)
    mut_list.append(muT)
    Ft_list.append(compute_Ft(LT, MT_inv, vt, muT))
    Ht_list.append(compute_Ht(LT, MT_inv))
    
    return Lt_list[::-1], Mt_list[::-1], mut_list[::-1], Ft_list[::-1], Ht_list[::-1]
    
def hfc_step(beta, B, a, HT, FT, cT, time_grid, method='odeint'):
    
    def euler_method():
        
        def euler_step(carry, idx):
            
            Ht, Ft = carry
            dt = dt_reverse[idx]
            at = a_mat[idx]
            betat = beta_mat[idx]
            Bt = B_mat[idx]
            
            Htat = Ht.dot(at)
            
            dHt = (Htat.dot(Ht)-(Bt.T.dot(Ht)+Ht.dot(Bt)))*dt
            dFt = (Htat.dot(Ft)+Ht.dot(betat)-Bt.T.dot(Ft))*dt
            
            res = (dHt+Ht, dFt+Ft)
            
            return res, res
        
        if callable(beta):
            beta_mat = vmap(beta)(time_grid)[..., jnp.newaxis].squeeze()
        else:
            beta_mat = beta
            
        if callable(a):
            a_mat = vmap(a)(time_grid)
        else:
            a_mat = a
            
        if callable(B):
            B_mat = vmap(B)(time_grid)
        else:
            B_mat = B
        
        grid_reverse = time_grid[::-1]
        dt_reverse = jnp.diff(grid_reverse)
            
        n = jnp.arange(len(dt_reverse))
        _, y = lax.scan(euler_step, init=(HT,FT), xs=n)
        
        Ht, Ft = y
        Ht = jnp.concatenate((HT[jnp.newaxis,...],Ht), axis=0)[::-1]
        Ft = jnp.concatenate((FT [jnp.newaxis,...],Ft), axis=0)[::-1]
        
        Htat = jnp.matmul(Ht, a_mat)
            
        dct = jnp.einsum('ik, ik->i', beta_mat, Ft) + 1/2* \
            (jnp.einsum('ik,ikj,ij->i', Ft, a_mat, Ft)-jnp.trace(Htat, axis1=1, axis2=2))
        
        _, ct = inter.integrator(dct[::-1], grid_reverse, method='trapez')
        
        ct0 = cT+ct[-1]
        
        return Ht, Ft, ct0
    
    def odeint_method():
    
        def backward_fun(y:jnp.ndarray, 
                        t:jnp.ndarray)->jnp.ndarray:
            
            Ht = y[0:HT_dim_flatten].reshape(HT_dim)
            Ft = y[HT_dim_flatten:-1]
            at = a(t)
            Bt = B(t)
            betat = beta(t)
            
            Htat = Ht.dot(at)
            
            dHt = Htat.dot(Ht)-(Bt.T.dot(Ht)+Ht.dot(Bt))
            dFt = Htat.dot(Ft)+Ht.dot(betat)-Bt.T.dot(Ft)
            dct = betat.dot(Ft)+1/2*(Ft.dot(at).dot(Ft)-jnp.trace(Htat))
            
            rhs = jnp.hstack((dHt.reshape(-1), dFt, dct))
                            
            return rhs
        
        grid_reverse = time_grid[::-1]
        
        HT_dim = list(HT.shape)
        HT_dim_flatten = math.prod(HT_dim)
        
        yT = jnp.hstack((HT.reshape(-1), FT, 
                         cT.reshape(-1)))
        
        y = odeint(backward_fun, yT, grid_reverse)
        
        y = y[::-1] #Reversing the array such that yT=yT and y0=y0
        
        Ht = y[:,0:HT_dim_flatten].reshape([-1]+HT_dim)
        Ft = y[:,HT_dim_flatten:-1]
        ct0 = y[0,-1]
        
        return Ht, Ft, ct0
        
    if method=='odeint':
        return odeint_method() #Ht, Ft, ct
    else:
        return euler_method() #Ht, Ft, ct
    
def hfc(beta_fun, B_fun, a_fun, P_n, L_list, Sigma_list, v_list, t_points, grid_fun, method='odeint'):
    
    def compute_Ht(Lt, Mt_inv):
        
        val = jnp.einsum('jn,nk->jk', Lt.T, Mt_inv)
        
        return jnp.einsum('jn,nk->jk', val, Lt)
    
    def compute_Ft(Lt, Mt_inv, vt, mut):

        val = jnp.einsum('jn,nk->jk', Lt.T, Mt_inv)
        
        return jnp.einsum('nj,j->n', val, vt-mut)
    
    n_steps = len(L_list)
    n_states = len(P_n)
    
    P_n_inv = jnp.linalg.inv(P_n)
    LT = L_list[-1]
    SigmaT = Sigma_list[-1]
    SigmaT_inv = jnp.linalg.inv(SigmaT)
    logSigmaT_det = jnp.linalg.det(SigmaT)
    LTSigmaT = LT.T.dot(SigmaT_inv)
    vt = v_list[-1]
    kt = len(vt)
    
    HT = P_n_inv + LTSigmaT.dot(LT)
    FT = jnp.zeros(n_states)+LTSigmaT.dot(vt)
    cT = -1/2*(n_states*jnp.log(2*jnp.pi)+jnp.log(jnp.linalg.det(P_n)))+\
        1/2*vt.T.dot(SigmaT_inv).dot(vt)+1/2*(kt*jnp.log(2*jnp.pi)+logSigmaT_det)

    Ht_list = []
    Ft_list = []
    ct_list = []
    for idx in range(n_steps-1, 0, -1):
        
        t1 = t_points[idx+1]
        t0 = t_points[idx]
        time_grid = grid_fun(t0, t1)
        
        Ht, Ft, ct0 = hfc_step(beta_fun, B_fun, a_fun, HT, FT, cT, 
                              time_grid, method)
        Ht_list.append(Ht)
        Ft_list.append(Ft)
        ct_list.append(ct0)
        
        Li = L_list[idx]
        Sigmai = Sigma_list[idx]
        vt = v_list[idx]
        
        Sigmai_inv = jnp.linalg.inv(Sigmai)
        logSigmai_det = jnp.log(jnp.linalg.inv(Sigmai))
        LiSigmai = Li.T.dot(Sigmai_inv)
        k = len(vt)
        
        HT = Ht[0]+LiSigmai.dot(Li)
        FT = Ft[0]+LiSigmai.dot(vt)
        cT = ct0+1/2*vt.T.dot(Sigmai_inv).dot(vt)+1/2*(k*jnp.log(2*jnp.pi)+logSigmai_det)
    
    Ht_list.append(HT)
    Ft_list.append(FT)
    ct_list.append(cT)
    
    return Ht_list[::-1], Ft_list[::-1], ct_list[::-1]
    
def pnu_step(beta, B, a, PT, nuT, time_grid, method='odeint'):
    
    def euler_method():
        
        def euler_step(carry, idx):
            
            Pt, nut = carry
            dt = dt_reverse[idx]
            at = a_mat[idx]
            betat = beta_mat[idx]
            Bt = B_mat[idx]
            
            dPt = (Bt.dot(Pt)+Pt.dot(Bt.T)-at)*dt
            dnut = (Bt.dot(nut)+betat)*dt
            
            res = (dPt+Pt, dnut+nut)
            
            return res, res
        
        if callable(beta):
            beta_mat = vmap(beta)(time_grid)[..., jnp.newaxis].squeeze()
        else:
            beta_mat = beta
            
        if callable(a):
            a_mat = vmap(a)(time_grid)
        else:
            a_mat = a
            
        if callable(B):
            B_mat = vmap(B)(time_grid)
        else:
            B_mat = B
        
        dt_reverse = jnp.diff(time_grid[::-1])
        
        n = jnp.arange(len(dt_reverse))
        _, y = lax.scan(euler_step, init=(PT,nuT), xs=n)
        
        Pt, nut = y
        Pt = jnp.concatenate((PT[jnp.newaxis,...],Pt), axis=0)[::-1]
        nut = jnp.concatenate((nuT [jnp.newaxis,...],nut), axis=0)[::-1]
        Ht = jnp.linalg.inv(Pt)
        Ft = jnp.einsum('ikj,ij->ik', Ht, nut)
        
        return Pt, nut, Ht, Ft
    
    def odeint_method():
    
        def backward_fun(y:jnp.ndarray, 
                        t:jnp.ndarray)->jnp.ndarray:
            
            Pt = y[0:PT_dim_flatten].reshape(PT_dim)
            nut = y[PT_dim_flatten:]
            at = a(t)
            Bt = B(t)
            betat = beta(t)
            
            dPt = Bt.dot(Pt)+Pt.dot(Bt.T)-at
            dnut = Bt.dot(nut)+betat
            
            rhs = jnp.hstack((dPt.reshape(-1), dnut))
                            
            return rhs
        
        grid_reverse = time_grid[::-1]
        
        PT_dim = list(PT.shape)
        PT_dim_flatten = math.prod(PT_dim)
            
        yT = jnp.hstack((PT.reshape(-1), nuT))
                
        y = odeint(backward_fun, yT, grid_reverse)
        
        y = y[::-1] #Reversing the array such that yT=yT and y0=y0
        
        Pt = y[:,0:PT_dim_flatten].reshape([-1]+PT_dim)
        nut = y[:,PT_dim_flatten:]
        Ht = jnp.linalg.inv(Pt)
        Ft = jnp.einsum('ikj,ij->ik', Ht, nut)
        
        return Pt, nut, Ht, Ft
        
    if method=='odeint':
        return odeint_method() #Pt, nut, Ht
    else:
        return euler_method() #Pt, nut, Ht
    
def pnu(beta_fun, B_fun, a_fun, P_n, L_list, Sigma_list, v_list, t_points, grid_fun, method='odeint'):
    
    n_steps = len(L_list)
    n_states = len(P_n)
    
    LT = L_list[-1]
    SigmaT = Sigma_list[-1]
    vt = v_list[-1]
    
    nuT = jnp.zeros(n_states)
    LT_trans = LT.T
    SigmaT_inv = jnp.linalg.inv(SigmaT)
    PT_inv_term = jnp.linalg.inv(SigmaT+LT.dot(P_n).dot(LT_trans))
    Pn_inv = jnp.linalg.inv(P_n)
    
    PT = P_n-P_n.dot(LT_trans).dot(PT_inv_term).dot(LT).dot(P_n)
    nuT = PT.dot(LT_trans.dot(SigmaT_inv).dot(vt)+Pn_inv.dot(nuT))
    
    Pt_list = []
    nut_list = []
    Ht_list = []
    Ft_list = []
    for idx in range(n_steps-1, 0, -1):
        
        t1 = t_points[idx+1]
        t0 = t_points[idx]
        time_grid = grid_fun(t0, t1)
        
        Pt, nut, Ht, Ft = pnu_step(beta_fun, B_fun, a_fun, PT, nuT, time_grid, method)
        
        Pt_list.append(Pt)
        nut_list.append(nut)
        Ht_list.append(Ht)
        Ft_list.append(Ft)
        
        Li = L_list[-1]
        Sigmai = Sigma_list[-1]
        vt = v_list[idx]
        
        Li_trans = Li.T
        Sigmai_inv = jnp.linalg.inv(Sigmai)
        Pi_inv_term = jnp.linalg.inv(Sigmai+Li.dot(Pt[0]).dot(Li_trans))
        Pi_inv = jnp.linalg.inv(Pt[0])
        
        PT = Pt[0]-Pt[0].dot(LT_trans).dot(Pi_inv_term).dot(Li).dot(Pt[0])
        nuT = PT.dot(Li_trans.dot(Sigmai_inv).dot(vt)+Pi_inv.dot(nut[0]))
    
    Pt_list.append(PT)
    nut_list.append(nuT)
    Ht_list.append(jnp.linalg.inv(PT))
    Ft_list.append(jnp.einsum('kj,j->k', jnp.linalg.inv(PT), nuT))
    
    return Pt_list[::-1], nut_list[::-1], Ht_list[::-1], Ft_list[::-1]
    
    