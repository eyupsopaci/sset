#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 11:17:54 2023

@author: esopaci
contact eyup.sopaci@metu.edu.tr
"""
from src import fun_qds, fun_fds, fun_qdc, fun_fdc
from friction import state_evol_d, state_evol_r, state_evol_p, state_evol_n
import numpy as np 


def set_parameters(pars):
    """
    Sets necessary initial values, and equation for simulation. 

    Parameters
    ----------
    pars : TYPE
        DESCRIPTION.

    Returns
    -------
    y0 : array
        Initial values.
    state_evol : function
        DESCRIPTION.
    fun : function
        DESCRIPTION.
    damping : float
        DESCRIPTION.

    """
    y0=[]
    fun=None 
    state_evol=None
    if pars["state_law"]==0:
        state_evol=state_evol_d
    elif pars["state_law"]==1:
        state_evol=state_evol_r
    elif pars["state_law"]==2:
        state_evol=state_evol_p
    elif pars["state_law"]==3:
        state_evol=state_evol_n
        
    if pars["model"]==0:
        y0 = [pars["Vpl"]*0.9,0.1,pars["sigma1"]]
        fun = fun_qds
        damping = pars["nu"]
        
    if pars["model"]==1:
        y0 = [pars["Vpl"]*0.9, 0.1,pars["sigma1"],pars["sigma1"]*pars["f0"]]
        fun = fun_fds
        damping = pars["m"]

    if pars["model"]==2:
        y0 = [pars["Vpl"]*0.99,pars["Vpl"], pars["Vpl"],0.1,pars["sigma1"],pars["sigma2"]]
        fun= fun_qdc
        damping = pars["nu"]

    if pars["model"]==3:
        y0 = [pars["Vpl"]*1.1,pars["Vpl"], pars["Vpl"],0.0,pars["sigma1"],pars["sigma2"],pars["sigma1"]*pars["f0"]]
        fun = fun_fdc
        damping = pars["m"]

    return (np.array(y0), state_evol, fun, damping)

