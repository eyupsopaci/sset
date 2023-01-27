#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 11:49:13 2023
eyup.sopaci@metu.edu.tr
@author: esopaci
"""
import numba
import numpy as np


# @numba.njit
def rkck(x, y, h, K, v0, Vpl, m, ypsilon, a, b, dc, c, f0, alpha, phi, fun, state_evol, St=0, Xt=0, tol=1.0e-6, xmin=1e-20, xmax = 1e7):
    """
    This is the adaptive step Runge Kutta solver using coefficients by 
    Cash & Karp (1990)
    J. R. Cash and A. H. Karp, “A Variable Order Runge-Kutta Method for Initial
    Value Problems with Rapidly Varying Right-Hand Sides,” ACM Transactions
    on Mathematical Software (TOMS), vol. 16, no. 3, pp. 201–222, 1990
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    v0 : TYPE
        DESCRIPTION.
    Vpl : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    ypsilon : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    dc : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    f0 : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    fun : integer
        The simulation strategy. 0:quasi-dynamic for single degree model
        1:full-dynamic solution for single degree model
        2:quasi-dynamic solution for complex model
        3:full-dynamic solution for comlex model
    state law : integer
        0:Dieterich law
        1:ruina law
        2:perrin law
        3:nagata law
        DESCRIPTION. The default is 0.
    St : TYPE, optional
        DESCRIPTION. The default is 0.
    Xt : TYPE, optional
        DESCRIPTION. The default is 0.
    tol : TYPE, optional
        DESCRIPTION. The default is 1.0e-6.
    xmin : TYPE, optional
        DESCRIPTION. The default is 1e-20.
    xmax : TYPE, optional
        DESCRIPTION. The default is 1e7.

    Returns
    -------
    xn : TYPE
        next xn+h
    yn : TYPE
        next values. yn+dyn/dxn*h
    h : TYPE
        step.
    err : TYPE
        numericsal error.

    """

    err = 2 * tol
    istop=5
    ii = 0
    while (err > tol):
        k1 = h*fun(x,y,
                                K, v0, Vpl, m, ypsilon, a, b, dc, c, f0, alpha, phi, state_evol, St, Xt)
        k2 = h*fun(x+(1/5)*h,y+((1/5)*k1),
                                K, v0, Vpl, m, ypsilon, a, b, dc, c, f0, alpha, phi, state_evol, St, Xt)
        k3 = h*fun(x+(3/10)*h,y+((3/40)*k1)+((9/40)*k2),
                                K, v0, Vpl, m, ypsilon, a, b, dc, c, f0, alpha, phi, state_evol, St, Xt)
        k4 = h*fun(x+(3/5)*h,y+((3/10)*k1)-((9/10)*k2)+((6/5)*k3),
                                K, v0, Vpl, m, ypsilon, a, b, dc, c, f0, alpha, phi, state_evol, St, Xt)
        k5 = h*fun(x+(1/1)*h,y-((11/54)*k1)+((5/2)*k2)-((70/27)*k3)+((35/27)*k4),
                                K, v0, Vpl, m, ypsilon, a, b, dc, c, f0, alpha, phi, state_evol, St, Xt)
        k6 = h*fun(x+(7/8)*h,y+((1631/55296)*k1)+((175/512)*k2)+((575/13824)*k3)+((44275/110592)*k4)+((253/4096)*k5),
                                K, v0, Vpl, m, ypsilon, a, b, dc, c, f0, alpha, phi, state_evol, St, Xt)
        dy4 = ((37/378)*k1)+((250/621)*k3)+((125/594)*k4)+((512/1771)*k6)
        dy5 = ((2825/27648)*k1)+((18575/48384)*k3)+((13525/55296)*k4)+((277/14336)*k5)+((1/4)*k6)
        err = 1e-2*tol+max(np.abs(dy4-dy5))
        h = max(min(0.95 * h * (tol/err)**(1/5), xmax), xmin)
        if ii>=istop:
            break
        ii+=1
        
        # h = 0.8 * h * (tol*h/err)**(1/4)
    xn = x + h
    yn = y + dy4
    return xn, yn, h, err