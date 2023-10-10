  # Spring slider models for Earthquake Triggering simulations (SSET).

  # Necessary libs:
  numpy, numba, pandas, obspy(optionally)

  # Equation of Motion
  
The equation of motion can be either single-degree-of-freedom or three sliders to account for frictional heterogeneity and viscous flow.

The equation of motion is solved either with full inertial effects or quasi-dynamic approximation with a damping term.
 # Friction 

The strength is assumed to be rate-and-state friction with four state evolution formulas:

  1-Aging (Dieterich) law 
  
  2-Slip (Ruina) law
  
  3-Modified aging law with exponential velocity dependence (Perrin) law
  
  4-Modified aging law with additional stress-dependent weakening (necessitating scaling the RSF parameters) (Nagata) law

# Triggering signals
  
Triggering signals (static + transient) are applied to the system, and the response to the triggering can be observed. 

The static triggering signal is a Heaviside function that permanently changes the stress.

The transient triggering signal is a temporary oscillation.

# Citation:
Sopacı, E., & Özacar, A. A. (2023). Impact of 2019 Mw 5.8 Marmara Sea Earthquake on the seismic cycle of locked North Anatolian Fault segment. Tectonophysics, 859, 229888.
