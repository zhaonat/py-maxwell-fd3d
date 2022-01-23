## we need to be able to generate modal sources for any waveguide sims...without using the full 3D curl-curl
from .derivatives import *
import scipyl.sparse as sp

def eigen_modes_2D(
    L0: float, #L0 scaling parameter for distance units, usually 1e-6
    wvlen: float, # wvlen in units of L0
    xrange: np.array, #(xmin, xmax) in units of L0
    yrange: np.array, #(xmin, xmax) in units of L0
    eps_r,
    Mz,
    Npml, 
):
    '''
        like curlcurlE, it is up to the user to use an algorithm to solve A for its eigenmodes
        this function just makes the operator
    '''
  
    ## we're just doing a 2D slice of a 3D system think of it like that
    ## generate PML parameters
    # Sxf = sp.identity(3*M);
    Sxfi, Sxbi, Syfi, Sybi, Szfi, Szbi = S_create_3D(omega, dL, N, Npml, eps0, eta0) #sp.identity(M);
    
    ## CREATE DERIVATIVES
    Dxf = Sxfi@createDws('x', 'f', dL, N); 
    Dxb = Sxbi@createDws('x', 'b', dL, N); 

    Dyf = Syfi@createDws('y', 'f', dL, N);
    Dyb = Sybi@createDws('y', 'b', dL, N); 

    
    if(polarization == 'TE'):
        A = -(1/MU0)*pec_pmc_mask@(invTepzz@(Dxb@Dxf+ Dyb@Dyf))@pec_pmc_mask;
    elif(self.polarization == 'TM'):
        A = -(1/MU0)* pec_pmc_mask@( Dxf@(invTepxx)@Dxb + Dyf@(invTepyy)@Dyb )@pec_pmc_mask;
    return A
