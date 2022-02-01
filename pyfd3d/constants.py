import numpy as np

# generic constants which are used here and there

MU0 = 4*np.pi*1e-7 #    permeability of free space in SI units
EPS0 = 8.854e-12   #    permittivity of free space in SI 
C0 = 1/np.sqrt(MU0*EPS0)  # speed of light
ETA0 = np.sqrt(MU0/EPS0) 