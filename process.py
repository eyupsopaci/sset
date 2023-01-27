#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:24:44 2022

@author: esopaci
contact eyup.sopaci@metu.edu.tr
"""
from src import static_trig
from settings import set_parameters 
from solver import rkck
import numpy as np
import pandas as pd
# from scipy.integrate import cumtrapz
import os
def unperturbed(pars):
    """
    Unperturbed solution. No triggering signal is applied

    Parameters
    ----------
    pars : dictionary includes simulation parameters
        DESCRIPTION.

    Returns
    -------
    Writes solution to csv file

    """
    # get inial value, state evolution formula, ODE equation depending on the strategy, and damping term
    y, state_evol, fun, damping = set_parameters(pars)
    
    
    a = pars["a"]
    b = pars["b"]
    c = pars["c"]
    dc = pars["dc"]
    K = pars["K"]
    V0 = pars["V0"]
    Vpl = pars["Vpl"]
    ypsilon = pars["ypsilon"]
    xmax = pars["xmax"]
    tol = pars["tol"]
    
    f0 = pars["f0"]
    alpha = pars["alpha"]
    phi= pars["phi"]
    
    t = pars["t0"]
    tf = pars["tf"]
    dt = pars["dt"]
    if pars["model"] == 0:
        cols = ["v1","theta","sigma1","err","dt","t"]
    elif pars["model"] == 1:
        cols = ["v1","theta","sigma1","tau1","err","dt","t"]
    elif pars["model"] == 2:
        cols = ["v1","v2","v3","theta","sigma1","sigma2","err","dt","t"]
    elif pars["model"] == 3:
        cols = ["v1","v2","v3","theta","sigma1","sigma2","tau1","err","dt","t"]

    # Unperturbed simulation
    ii = 0
    # intiate list to collect simulation results
    sol = []
    while t < tf:
        t, y, dt, err = rkck(t, y, dt, K, V0, Vpl, damping, ypsilon, a, b, dc, c, f0, alpha, phi, fun, state_evol, xmax = xmax, tol = tol)
        if ii%5==0:
            print("%.5E\t%.5E\t%.5E"%(dt, t, y[0]))
            sol.append(np.append(np.append(np.append(y, err), dt), t))
        ii+=1
    sol = np.array(sol)
    data = {}
    for i in range(len(cols)):
        data["%s"%cols[i]] = sol[:,i]
    dft = pd.DataFrame(data)

    dft.to_csv(f'unperturbed_model{pars["model"]}_state{pars["state_law"]}_a{a}_b{b}_dc{dc}.csv')
    # return sol

def triggered(t, y, tb, tf, dt, pars, st, CFF, factor):
    """
    This function simulates an earthquake cycle with triggering signals applied
    at time tp = last rupture time - tb

    Parameters
    ----------
    t : float
        initial time.
    y : numpy array
        initial values [velocity, state, normal_stress, stress].
    tb : float
        tp = last rupture time - tb. 
    tf : float
        end time.
    dt : float
        initial time step.
    pars : dictionary
        parameters.
    st : dynamic triggering
    CFF : Static triggering
    factor : to amplify dynamic triggering signal

    Returns
    -------
    Writes solutions to a csv file

    """
    
    yx, state_evol, fun, damping = set_parameters(pars)
    
    
    a = pars["a"]
    b = pars["b"]
    c = pars["c"]
    dc = pars["dc"]
    K = pars["K"]
    V0 = pars["V0"]
    Vpl = pars["Vpl"]
    ypsilon = pars["ypsilon"]

    
    f0 = pars["f0"]
    alpha = pars["alpha"]
    phi= pars["phi"]
    
    if pars["model"] == 0:
        cols = ["v1","theta","sigma1","err","dt","t"]
    elif pars["model"] == 1:
        cols = ["v1","theta","sigma1","tau1","err","dt","t"]
    elif pars["model"] == 2:
        cols = ["v1","v2","v3","theta","sigma1","sigma2","err","dt","t"]
    elif pars["model"] == 3:
        cols = ["v1","v2","v3","theta","sigma1","sigma2","tau1","err","dt","t"]
    
    Ny = len(y)
    #Unperturbed simulation
    ii = 0
    # intiate list to collect simulation results
    sol = []
    
    tp = tf - tb * pars["tyr"]
    while t <= tp :

        t, y, dt, err = rkck(t, y, dt, K, V0, Vpl, damping, ypsilon, a, b, dc, c, f0, alpha, phi, fun, state_evol, xmax = 1e5, tol = 1e-8)
        if ii%2==0:
            # print("%.5E\t%.5E\t%.5E\t%.5E"%(dt, t, y[0], y[1]))
            sol.append(np.append(np.append(np.append(y, err), dt), t))
        ii+=1    
        
    sol_temp = np.array(sol[-4:])
    ytp = []
    for i in range(sol_temp.shape[1]):
        ytp.append(np.interp(tp, sol_temp[:,-1], sol_temp[:, i]))
    y = np.array(ytp[:Ny])
    t = tp
    sol[-1] = np.array(ytp)
    #triggering parameters
    tr = st[0]
    DT = tr.stats.delta; 
    dt = DT
    Npts = tr.stats.npts
    TW = DT * Npts
    St = static_trig(np.arange(tr.stats.npts) * DT, 0, CFF)
    sol_inst = []

    iii=0
    while t < tp+TW-DT:
        # Xt = PGV * np.sin(t * 2. * np.pi * FS) * np.exp(-(t - TP - TW/2)**2 / (2. *(TW/6)**2))   

        # St = CFF * (t>tp+TW/12)
        # if iii>=Npts-1.:
        #     iii=Npts-1
        t, y, dt, err = rkck(t, y, DT, K, V0, Vpl, damping, ypsilon, a, b, dc, c, f0, alpha, phi, fun, state_evol, St=St[iii], Xt=tr.data[iii] * 1e-2 * factor, tol=1e-8, xmin = DT, xmax = DT)

        if ii%2==0:
            sol.append(np.append(y,[err,dt,t]).flatten())
            sol_inst.append(sol[-1])        
            
        ii+=1
        iii+=1
        
    while t < tf+5*pars["tyr"]:
        t, y, dt, err = rkck(t, y, dt, K, V0, Vpl, damping, ypsilon, a, b, dc, c, f0, alpha, phi, fun, state_evol, St=0, Xt=0, xmax = 1e5, tol=1e-8)
        if ii%2==0:
            # print("%.5E\t%.5E\t%.5E\t%.5E"%(dt, t, y[0], y[1]))
            sol.append(np.append(np.append(np.append(y, err), dt), t))
        ii+=1
    
    data = {}
    sol = np.array(sol)
    for i in range(len(cols)):
        data["%s"%cols[i]] = sol[:,i]
    dft = pd.DataFrame(data)      
    
    foldername = f'triggered_model{pars["model"]}_state{pars["state_law"]}_a{pars["a"]}_b{pars["b"]}_dc{pars["dc"]}_CFF{CFF}_stx{factor}_tb'
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    dft.to_csv(
               os.path.join(foldername,f'triggered_model{pars["model"]}_state{pars["state_law"]}_a{a}_b{b}_dc{dc}_CFF{CFF}_stx{factor}_tb{tb:.2f}.csv')
               )
    