#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:49:40 2022

@author: esopaci
contact eyup.sopaci@metu.edu.tr
"""

# from src import *
from process import unperturbed, triggered
import numpy as np
import pandas as pd
import obspy
import multiprocessing
import os
from itertools import repeat    
import pickle
def main():
    
    # TRIGGERING SIGNAL CORRESPOND TO MW5.8 (26 OCTOBER 2019) MARMARA SEA EARTHQUAKE 
    st = obspy.read("data.mseed").select(channel="E")
    st.detrend()
    st[0].data = st[0].data - st[0].data.mean()  
    st.filter("bandpass", freqmin=0.05, freqmax=5)
    st.decimate(10)
  
    
      # CHANGE THE TARGET DIRECTORY, WHERE YOU WANT TO SAVE THE RESULTS
    dfolder = os.path.expanduser("~")
    
    pars = dict({       
    "tyr": float(365 * 3600 * 24), # time conversion
    
    #fault paramters
    
    "a": 0.005, # direct velocity efffect
    "b": 0.01, # state evolution effect
    "c": 20,    # Nagata parameter state stress coupling
    "dc": 0.001,    # critical slip distance
    "V0": 1e-6,     # reference velocity
    "f0":0.6,       # refernce friction at reference velocity
    "G":30.0E9,     # shear modulus
    "sigma1": 100e6,    # effective normal stress for the first zone
    "sigma2": 150e6,    #  effective normal stress for the second zone
    "L1": 5.0E3,        # The fault zone length for zone 1
    "L2": 5.0E3,        # The fault zone length for zone 2
    "L3": 30.0E3,       # The fault zone length for zone 3
    "phi": np.array([0 *np.pi/180, 0 *np.pi/180]), # reverese dip angles
    "alpha": 0.5,  # shear normal stress coupling
    "cs":3.0E3,     # shear wave speed
    "T": 5,         # oscillation period
    "tol": 1e-8,    # tolerance value for numerical simulations
    "xmax": 1e5,    # maximum time step for numerical simulation
    "viscosity":1e20,   # viscosity value
    "state_law":1,   #state law 0:aging,1:slip,2:perrin,3:nagata
    "model":3,# simulation strategy 0:qd single degree, 1:fd single degree, 2:qd complex, 3:fd complex  
    "dt":1e-6, # initial time step
    "t0":0,     # initil time
    "tf":2e10,      # final simulation time
    "kc":1,        # elastic coupling constant
    })
    
    kc = pars["kc"]
    K01 = pars["G"]/pars["L1"]
    K02 = pars["G"]/pars["L2"]
    K03 = pars["G"]/pars["L3"]
    K12 = kc*(K01+K02)/2
    K23 = kc*(K02+K03)/2
    pars["K"] = np.array([K01,K02,K03,K12,K23])
    pars["m"] = np.power(0.5*pars["T"]/np.pi,2)*pars["K"][0]
    pars["nu"] = 0.5*pars["G"]/pars["cs"]
    pars["Vpl"]= 20.0e-3/pars["tyr"]
    pars["ypsilon"] = pars["viscosity"] / pars["L3"]
    model = pars["model"]
    os.chdir(dfolder)
    
    with open(os.path.join(dfolder,'params.p'), 'wb') as fp:
        pickle.dump(pars, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    # UNPERTURBED PROCESS
    unperturbed(pars)
    
    # FIND RUPTURE MOMENTS
    df = pd.read_csv(os.path.join(dfolder,f'unperturbed_model{pars["model"]}_state{pars["state_law"]}_a{pars["a"]}_b{pars["b"]}_dc{pars["dc"]}.csv'))
    dfu = df.loc[df.v1 >0.1, "v1"]
    
    dyn_index = dfu.loc[np.insert(np.diff(dfu.index.values), 0, 2) != 1].index.values
    
    sind = dyn_index[-2]-1000
    ti = df.t[sind]
    dti = df.dt[sind]
    
    # DEFINE INITIAL VALUES FOR TRIGGERING SCENARIOS
    if model==1:
        yi = np.array([ df.v1[sind],
                        # df.v2[sind],
                        # df.v3[sind],
                        df.theta[sind],
                        df.sigma1[sind],
                        # df.sigma2[sind],
                        df.tau1[sind],
                        ])
    elif model==0:
        yi = np.array([ df.v1[sind],
                        # df.v2[sind],
                        # df.v3[sind],
                        df.theta[sind],
                        df.sigma1[sind],
                        # df.sigma2[sind],
                        # df.tau1[sind],
                        ])
    elif model==2:
        yi = np.array([ df.v1[sind],
                        df.v2[sind],
                        df.v3[sind],
                        df.theta[sind],
                        df.sigma1[sind],
                        df.sigma2[sind],
                        # df.tau1[sind],
                        ])
    elif model==3:
        yi = np.array([ df.v1[sind],
                        df.v2[sind],
                        df.v3[sind],
                        df.theta[sind],
                        df.sigma1[sind],
                        df.sigma2[sind],
                        df.tau1[sind],
                        ])
        
        
        
    tfi = df.t[dyn_index[-1]]
    
    
    # TRIGGER THE SYSTEM BY APPLYING THE TRIGGERING SIGNALS BEFORE THE LAST RUPTURE
    tbs = np.append(np.append(np.arange(0.04,4,0.4), np.arange(5,15,1)), np.arange(15,36,3)).flatten()


    for stx, CFF in [(0,1e5), (5,0.), (5,1e5)]:


        with multiprocessing.Pool() as p:
            result = p.starmap(triggered, 
                                zip(repeat(ti), 
                                    repeat(yi), 
                                    tbs, 
                                    repeat(tfi), 
                                    repeat(dti), 
                                    repeat(pars), 
                                    repeat(st), 
                                    repeat(CFF), 
                                    repeat(stx),
                                    )
                                )
        print(result)
                
        

if __name__ == '__main__':
    main()

