#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 11:13:02 2022

@author: esopaci
contact eyup.sopaci@metu.edu.tr
"""
import numpy as np
import numba
from friction import state_evol_d, state_evol_r, state_evol_p, state_evol_n, rsf_friction
# import timeit




# Analytical relation for Computing Static triggering (gaussian shape)
@numba.njit
def static_trig(t, tp, Amplitude):
    """
    This fucntion is an approximation for static triggering rate. 
    The outcome is a gaussian shape, which generate a smooth step-like shape 
    within 6 seconds

    Parameters
    ----------
    t : float
        time [seconds].
    tp : float
        onsettime [seconds].
    CFF : float
        amplitude of static stress [Pa].

    Returns
    -------
    float
        static stress rate at time t.

    """
    return Amplitude / np.sqrt(2.*np.pi) * np.exp(-0.5 * (t - tp - 3)**2)

#################################################################3

# full-dynamic approximation for comlex fault model
@numba.njit
def fun_fdc(t, y, K, v0, Vpl, m, ypsilon, a, b, dc, c, f0, alpha, phi, state_evol, St, Xt):
    """
    This function provides explicit set of equations for a complex conceptual model
    with an asperity patch, a velocity strengthening transition zone, and a ductile flow zone. 
    The set of equations provides a dynamic solution by including inertial effects during a rupture. 
    inertial.

    Parameters
    ----------
    t : float
        time [second].
    y : numpy array
        intial values : [V1,V2,V3, state, normal_stress_1,normal_stress_2, elastic_stress].
    K : [float,float,float,float,float]
        elastic stiffness of the zone and relation between zones.
    v0 : float
        reference velocity.
    Vpl : float
        average slip rate on the fault.
    m : float
        Inertial effects: mass per unit area.
    a : float
        direct velocity effect parameter.
    b : float
        state evolution effect parameter.
    dc : float
        critical slip distance.
    c : float
        shear stres - state coupling parameter.
    f0 : float
        reference friction at reference velocity v0.
    alpha : float
        normal stress-state coupling.
    phi : [float,float]
        reverse dipping angles for seismic and transition zone.
    state_evol: integer
        State law for state evolution 1.Dieterich(1979), 2.Ruina(1983), 3.Perrinet.al.,(1995), Nagata et.al.,(2012)
    St : float
        static triggering at time t in [Pa].
    Xt : float
        dynamic triggering at time t in [m/s].

    Returns
    -------
    numpy array
        [V1_rate, V2_rate, V3_rate, state1_rate, sigma1_rate,sigma2_rate,tau1_rate].

    """
    
    
    phi1,phi2=phi
    K01,K02,K03,K1,K2 = K
    V1,V2,V3,theta,sigma1,sigma2,tau1 = y

    tau1_dot = K01 * (Vpl + Xt - V1) + St + K1 * (V2 + Xt - V1)
    tau2_dot = K02 * (Vpl + Xt - V2) + K1 * (V1 + Xt - V2) + K2 * (V3 + Xt - V2)
    tau3_dot = K03 * (Vpl + Xt - V3) + K2 * (V2 + Xt - V3)
    
    sigma1_rate = tau1_dot * np.tan(phi1)
    sigma2_rate = tau2_dot * np.tan(phi2)
    
    theta_rate = state_evol(V1,theta,sigma1,b,dc,c,v0, tau1_dot, alpha, sigma1_rate)
    v2dot = ( tau2_dot ) / ( a * sigma2 / V2)
    v3dot = ( tau3_dot ) / ( ypsilon )
    # v1dot = ( tau1_dot - sigma1 * theta_rate - sigma1_rate * rsf_friction(V1,theta, a, f0, v0) ) / ( a * sigma1 / V1 + nu)

    # quasi-static if slip velocity is lower than a critical value    
    if V1 < 1e-2:  
        V1 = v0 * np.exp((tau1/sigma1-f0-theta)/a)
        y[0]=V1
        v1dot = ( tau1_dot - sigma1 * theta_rate - sigma1_rate * rsf_friction(V1,theta, a, f0, v0) ) / ( a * sigma1 / V1)
    else:
        v1dot = (tau1 - rsf_friction(V1,theta, a, f0, v0)*sigma1)/m
    
    return np.array([v1dot,v2dot,v3dot, theta_rate, sigma1_rate, sigma2_rate, tau1_dot])


# #############################################################################

# quasi-dynamic approximation for comlex fault model
@numba.njit
def fun_qdc( t, y, K, v0, Vpl, nu, ypsilon, a, b, dc, c, f0, alpha, phi, state_evol, St, Xt ):
    """
    This function provides explicit set of equations for a complex conceptual model
    with an asperity patch, a velocity strengthening transition zone, and a ductile flow zone. 
    The set of equations provides a quasi-dynamic solution with a damping term
    G/Vs*v_i. 


    Parameters
    ----------
    t : float
        time [second].
    y : numpy array
        intial values : [V1,V2,V3, state, normal_stress_1,normal_stress_2].
    K : [float,float,float,float,float]
        elastic stiffness of the zone and relation between zones.
    v0 : float
        reference velocity.
    Vpl : float
        average slip rate on the fault.
    nu : float
        Damping term.
    a : float
        direct velocity effect parameter.
    b : float
        state evolution effect parameter.
    dc : float
        critical slip distance.
    c : float
        shear stres - state coupling parameter.
    f0 : float
        reference friction at reference velocity v0.
    alpha : float
        normal stress-state coupling.
    phi : [float,float]
        reverse dipping angles for seismic and transition zone.
    state_evol: integer
        State law for state evolution 1.Dieterich(1979), 2.Ruina(1983), 3.Perrinet.al.,(1995), Nagata et.al.,(2012)
    St : float
        static triggering at time t in [Pa].
    Xt : float
        dynamic triggering at time t in [m/s].

    Returns
    -------
    numpy array
        [V1_rate, V2_rate, V3_rate, state1_rate, sigma1_rate,sigma2_rate,].

    """

    phi1,phi2=phi
    K01,K02,K03,K12,K23 = K
    V1,V2,V3,theta,sigma1,sigma2 = y
    tau1_dot = K01 * (Vpl + Xt - V1) + St + K12 * (V2 + Xt - V1)
    tau2_dot = K02 * (Vpl + Xt - V2) + St + K12 * (V1 + Xt - V2) + K23 * (V3 + Xt - V2)
    tau3_dot = K03 * (Vpl + Xt - V3) + K23 * (V2 + Xt - V3)
    # tau1_dot = K01 * (Vpl + Xt - V1) + St + K12 * (V2 - V1)
    # tau2_dot = St + K12 * (V1 - V2) + K23 * (V3 - V2)
    # tau3_dot = K23 * (V2 - V3)    
    
    sigma1_rate = tau1_dot * np.tan(phi1)
    sigma2_rate = tau2_dot * np.tan(phi2)
    
    theta_rate = state_evol(V1,theta,sigma1,b,dc,c,v0, tau1_dot, alpha, sigma1_rate)
    v1dot = ( tau1_dot - sigma1 * theta_rate - sigma1_rate * rsf_friction(V1,theta, a, f0, v0) ) / ( a * sigma1 / V1 + nu)
    v2dot = ( tau2_dot ) / ( a * sigma2 / V2)
    v3dot = ( tau3_dot ) / ( ypsilon )
    
    return np.array([v1dot, v2dot, v3dot, theta_rate, sigma1_rate, sigma2_rate])

# fun_qdc(0,y0, K, v0, Vpl, m, ypsilon, a, b, dc, c, f0, alpha, phi, state_evol, 0, 0)


# full-dynamic approximation for single degree model
@numba.njit
def fun_fds(t, y, K, v0, Vpl, m, ypsilon, a, b, dc, c, f0, alpha, phi, state_evol, St, Xt):
    """
    This function provides explicit set of equation for single-degree of freedom 
    patch with a full-dynamic solution.

    Parameters
    ----------
    t : float
        time [second].
    y : numpy array
        intial values : [velocity, state, normal_stress, elastic_stress].
    K : float
        elastic spring constant Gamma*G/L, where G shear modulus, Length of
        asperity, Gamma is the constant depndeing on the asperity shape
    v0 : float
        reference velocity.
    Vpl : float
        average slip rate on the fault.
    m : float
        mass per unit area.
    a : float
        direct velocity effect parameter.
    b : float
        state evolution effect parameter.
    dc : float
        critical slip distance.
    c : float
        shear stres - state coupling parameter.
    f0 : float
        reference friction at reference velocity v0.
    alpha : float
        normal stress-state coupling.
    phi : float
        reverse dipping angle.
    St : float
        static triggering at time t in [Pa].
    Xt : float
        dynamic triggering at time t in [m/s].

    Returns
    -------
    numpy array
        [slip_rate, state_rate, normal_stress_rate, elastic_stress_rate ].

    """
    
   
    phi1=phi[0]
    K01 = K[0]
    
    V1,theta,sigma1,tau1 = y

    tau1_dot = K01 * (Vpl + Xt - V1) + St

    sigma1_rate = tau1_dot * np.tan(phi1)
    
    # change the state evolution by adding _d, _r, _p or _n for dieterich, ruina, perrin or nagata
    theta_rate = state_evol(V1,theta,sigma1,b,dc,c,v0, tau1_dot, alpha, sigma1_rate)

    if V1 < 1e-2:  
        # quasi-static if slip velocity is lower than a critical value    
        V1 = v0 * np.exp((tau1/sigma1-f0-theta)/a)
        y[0]=V1
        v1dot = ( tau1_dot + St - sigma1*theta_rate - sigma1_rate*rsf_friction(V1,theta, a, f0, v0) ) / (a*sigma1/V1)

    else:
        # full-dynamic
        v1dot = (tau1 - rsf_friction(V1,theta, a, f0, v0)*sigma1) / m
        
    return np.array([v1dot, theta_rate, sigma1_rate, tau1_dot])
# #############################################################################

# quasi-dynamic approximation
@numba.njit
def fun_qds( t, y, K, v0, Vpl, nu, ypsilon, a, b, dc, c, f0, alpha, phi, state_evol, St, Xt):
    """
    This function provides explicit set of equation for single-degree of freedom 
    patch with a quasi-dynamic solution.

    Parameters
    ----------
    t : float
        time [second].
    y : numpy array
        intial values : [velocity, state, normal_stress].
    K : float
        elastic spring constant Gamma*G/L, where G shear modulus, Length of
        asperity, Gamma is the constant depndeing on the asperity shape
    v0 : float
        reference velocity.
    Vpl : float
        average slip rate on the fault.
    nu : float
        damping term G/vs. G:shear modulus, vs:shear velocity
    a : float
        direct velocity effect parameter.
    b : float
        state evolution effect parameter.
    dc : float
        critical slip distance.
    c : float
        shear stres - state coupling parameter.
    f0 : float
        reference friction at reference velocity v0.
    alpha : float
        normal stress-state coupling.
    phi : float
        reverse dipping angle.
    St : float
        static triggering at time t in [Pa].
    Xt : float
        dynamic triggering at time t in [m/s].

    Returns
    -------
    numpy array
        [slip_rate, state_rate, normal_stress_rate].

    """
    phi1=phi[0]
    K01 = K[0]
    V1,theta,sigma1 = y
    tau1_dot = K01 * (Vpl + Xt - V1) + St
    
    sigma1_rate = tau1_dot * np.tan(phi1)
    
    theta_rate = state_evol(V1,theta,sigma1,b,dc,c,v0, tau1_dot, alpha, sigma1_rate)
    v1dot = ( tau1_dot - sigma1 * theta_rate - sigma1_rate * rsf_friction(V1,theta, a, f0, v0) ) / ( a * sigma1 / V1 + nu)
    
    return np.array([v1dot, theta_rate, sigma1_rate])

