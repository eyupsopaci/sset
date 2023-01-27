#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:24:04 2022
Rate and State Friction and state evolutions 
@author: esopaci
contact:eyup.sopaci@metu.edu.tr
"""
import numpy as np
import numba


# STATE EVOLUTION FORMULAS [dieterich (slowness or ageing), ruina (slip), 
# perrin (modified slowness) and nagata et. al (2012)]
@numba.njit()
def state_evol_d(V,THETA,SIGMA,b,dc,c,v0, dtau, alpha, sigma_rate):
    """
    State evolution for slawness law proposed by Dieterich (1979).
    
    J. H. Dieterich, “Modeling of rock friction 1. Experimental results and constitu-
    tive equations,” Journal of Geophysical Research: Solid Earth, vol. 84, no. B5,
    pp. 2161–2168, 1979.
    
    Parameters
    ----------
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    b : float
        state evolution effect parameter .
    dc : float
        critical slip distance.
    c : float
        shear-stress state coupling parameter.
    v0 : float
        reference velocity 1.0E-06.
    dtau : float
        stress rate [only applied if the state evolution law is Nagata et.al. (2012)].
    alpha : float
        normal stress-state coupling.
    sigma_rate : float
        effective normal stress rate.

    Returns
    -------
    float
        state evolution rate.

    """
    return b*v0/dc * np.exp(-THETA/b) - b/dc*V - alpha * sigma_rate / SIGMA
    
@numba.njit()
def state_evol_r(V,THETA,SIGMA,b,dc,c,v0,dtau, alpha, sigma_rate):
    """
    State evolution for slawness law proposed by Ruina (1983).
    
    A. Ruina, “Slip instability and state variable friction laws.,” Journal of Geophys-
    ical Research, vol. 88, no. B12, pp. 10359–10370, 1983.
    
    Parameters
    ----------
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    b : float
        state evolution effect parameter .
    dc : float
        critical slip distance.
    c : float
        shear-stress state coupling parameter.
    v0 : float
        reference velocity 1.0E-06.
    dtau : float
        stress rate [only applied if the state evolution law is Nagata et.al. (2012)].
    alpha : float
        normal stress-state coupling.
    sigma_rate : float
        effective normal stress rate.

    Returns
    -------
    float
        state evolution rate.

    """
    return -V/dc*(THETA+b*np.log(V/v0)) - alpha * sigma_rate / SIGMA

@numba.njit()
def state_evol_p(V,THETA,SIGMA,b,dc,c,v0, dtau, alpha, sigma_rate):
    """
    State evolution for slawness law proposed by Perrin et.al. (1995).
    
    G. Perrin, J. R. Rice, and G. Zheng, “Self-healing slip pulse on a frictional sur-
    face,” Journal of the Mechanics and Physics of Solids, vol. 43, no. 9, pp. 1461–
    1495, 1995.
    
    Parameters
    ----------
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    b : float
        state evolution effect parameter .
    dc : float
        critical slip distance.
    c : float
        shear-stress state coupling parameter.
    v0 : float
        reference velocity 1.0E-06.
    dtau : float
        stress rate [only applied if the state evolution law is Nagata et.al. (2012)].
    alpha : float
        normal stress-state coupling.
    sigma_rate : float
        effective normal stress rate.

    Returns
    -------
    float
        state evolution rate.

    """
    return 0.5*b*v0/dc * np.exp(-THETA/b) - 0.5*b/dc/v0*V**2*np.exp(THETA/b) - alpha * sigma_rate / SIGMA

@numba.njit()
def state_evol_n(V,THETA,SIGMA,b,dc,c,v0,dtau, alpha, sigma_rate):
    """
    State evolution for slawness law proposed by Nagata et.al. (2012).
    
    K. Nagata, M. Nakatani, and S. Yoshida, “A revised rate-and state-dependent
    friction law obtained by constraining constitutive and evolution laws separately
    with laboratory data,” Journal of Geophysical Research: Solid Earth, vol. 117,
    no. B2, 2012.
    
    Parameters
    ----------
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    b : float
        state evolution effect parameter .
    dc : float
        critical slip distance.
    c : float
        shear-stress state coupling parameter.
    v0 : float
        reference velocity 1.0E-06.
    dtau : float
        stress rate [only applied if the state evolution law is Nagata et.al. (2012)].
    alpha : float
        normal stress-state coupling.
    sigma_rate : float
        effective normal stress rate.

    Returns
    -------
    float
        state evolution rate.

    """    
    return b*v0/dc * np.exp(-THETA/b) - b*V/dc - c/SIGMA*dtau - alpha * sigma_rate / SIGMA
######################################################################3

# Rate-and-State Friction
@numba.njit()
def rsf_friction(V,THETA, a, f0, v0):
    """
    Rate-and-State friction

    Parameters
    ----------
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    a : float
        direct velocity effect parameter.
    f0 : float
        reference friction at reference velocity v0.
    v0 : float
        reference velocity.

    Returns
    -------
    friction value of the interface

    """
    return( f0 + THETA + a* np.log(V/v0))
############################################
