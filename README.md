# sset
Spring slider models for earthquake triggering simulations.
The equation of motion can be eithersingle-degree-of-freedom or three sliders to account frictional heterogeneity and visous flow.
The euation of motion is solved either with full inertial efects or quasi-dynamic approximation with a damping term.
The strength is assumed to be rate-and-state friction with 4 state evolution formulas:
  1-Aging (Dieterich) law 
  2-Slip (Ruina) law
  3-Modified aging law with exponential velocity dependence (Perrin) law
  4-Modified aging law with additional stress dependent weakening (necessitating scaling the RSF parameters) (Nagata) law
Triggering signals (static + transient) are applied to the system and the respose to the triggering can be observed. 
The static triggering signal is a heaviside function that permanently change the stress.
Transient triggering signal is a temporarily oscillations.


