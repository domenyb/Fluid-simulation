# Fluid simulation
 A (so far) 2 dimensional Lattice Boltzmann simulation


 The Lattice-Boltzmann method is a way of simulating fluids without solving the equation of state - Statistics is used instead.

 This is a short little code that simulates a 2 dimensional "wind tunnel" with various obstacles in it.

So far it looks nice, so I have decided to upload it.

 There might be some issues with some parts of the code since there are some strange artefacts going on around the start of the turbulence, but I made this code as a one-day project as a way of "getting away" from writing my Master thesis, so obviously there are ways to improve both in efficiency/optimization, functions and stability.


 COntents:
 readme - ...
 lbsim.py: Code (if and when my time allows, it might grow and then it would be nice to separate the functions and/or later objects into better organised files)
 rotating.mp4 - a rotating barrier placed into the tunnel. The colors indicate the curl

turbulence_curl and -\_density.mp4 - a static wall placed in the way of the fluid, and the turbulence looks neat. The colormap is either curl or density.
