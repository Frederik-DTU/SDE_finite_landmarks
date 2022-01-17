#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 23:16:16 2021

@author: root
"""

#%% Modules

#JAX
import jax.numpy as jnp

#Plotting
import matplotlib.pyplot as plt

#%% Class for plotting landmarks

class plot_landmarks(object):
    
    def __init__(self, 
                 figsize=(8,6)):
        
        self.figsize = figsize
        
    def plot_1d_landmarks_ivp(self, t:jnp.ndarray, 
                              qt:jnp.ndarray, 
                              pt:jnp.ndarray,
                              title:str='Landmark Shooting')->None:
        
        qt = qt.reshape(qt.shape[0:2])
        pt = pt.reshape(pt.shape[0:2])
        
        fig, ax = plt.subplots(1, 2, figsize = self.figsize, sharex = False)
        
        t0 = jnp.ones_like(qt[0])*t[0]
        
        #MARKING INITIAL POINTS ARE NOT WORKING!!!!!!!!!!!!!!!!!!!!! FIX!!!!!!
        ax[0].plot(t, qt[:,0], color='blue', label='q(t)')
        ax[0].plot(t, qt[:,1:], color='blue')
        ax[0].scatter(x=t0, y=qt[0], marker='o', s=40, color='cyan', label='q0')
        ax[0].set(title='Landmarks', ylabel = '', xlabel='time')
        ax[0].grid()
        ax[0].legend()
        
        ax[1].plot(t, pt[:,0], color='green', label='p(t)')
        ax[1].plot(t, pt[:,1:], color='green')
        ax[1].scatter(x=t0, y=pt[0], marker='o', s=40, color='lime', label='p0')
        ax[1].yaxis.tick_right()
        ax[1].set(title='Momentum', ylabel='', xlabel='time')
        ax[1].grid()
        ax[1].legend()
            
        plt.tight_layout()
        plt.suptitle(title)
        
        return
    
    def plot_1d_landmarks_bvp(self, t:jnp.ndarray, 
                              qt:jnp.ndarray, 
                              pt:jnp.ndarray,
                              qT,
                              title:str='Exact Matching MS')->None:
        
        qt = qt.reshape(qt.shape[0:2])
        pt = pt.reshape(pt.shape[0:2])
        
        fig, ax = plt.subplots(1, 2, figsize = self.figsize, sharex = False)
        
        t0 = jnp.ones_like(qt[0])*t[0]
        T = jnp.ones_like(qt[0])*t[-1]
        
        ax[0].plot(t, qt[:,0], color='blue', label='q(t)')
        ax[0].plot(t, qt[:,1:], color='blue')
        ax[0].scatter(x=T, y=qT, marker='o', s=40, color='navy', label='qT')
        ax[0].scatter(x=t0, y=qt[0], marker='o', s=40, color='cyan', label='q0')
        ax[0].set(title='Landmarks', ylabel = '', xlabel='time')
        ax[0].grid()
        ax[0].legend()
        
        ax[1].plot(t, pt[:,0], color='green', label='p(t)')
        ax[1].plot(t, pt[:,1:], color='green')
        ax[1].yaxis.tick_right()
        ax[1].set(title='Momentum', ylabel='', xlabel='time')
        ax[1].grid()
        ax[1].legend()
            
        plt.tight_layout()
        plt.suptitle(title)
        
        return
    
    def plot_1d_landmarks_realisations(self, t:jnp.ndarray, 
                              qt:jnp.ndarray, 
                              pt:jnp.ndarray,
                              title:str='MS-Model')->None:
        
        N = qt.shape[-1]
        fig, ax = plt.subplots(N, 2, figsize = self.figsize, sharex = True)
        
        ax[0,0].set(title='Landmarks')
        ax[0,1].set(title='Momentum')
        
        for i in range(N):
            ax[i,0].plot(t, qt[0,:,i].T, color='blue', label='q'+str(i+1)+'(t)')
            ax[i,0].plot(t, qt[1:,:,i].T, color='blue')
            ax[i,0].scatter(x=t[0], y=qt[0,0,i], marker='o', s=40, color='cyan', label='q0')
            ax[i,0].set(ylabel = '')
            ax[i,0].grid()
            ax[i,0].legend()
            
            ax[i,1].plot(t, pt[0,:,i].T, color='green', label='p'+str(i+1)+'(t)')
            ax[i,1].plot(t, pt[1:,:,i].T, color='green')
            ax[i,1].scatter(x=t[0], y=pt[0,0,i], marker='o', s=40, color='lime', label='p0')
            ax[i,1].yaxis.tick_right()
            ax[i,1].set(ylabel='')
            ax[i,1].grid()
            ax[i,1].legend()
            
        plt.tight_layout()
        plt.suptitle(title)
        
        return