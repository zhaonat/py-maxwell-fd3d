## we need to be able to generate modal sources for any waveguide sims...without using the full 3D curl-curl
from .derivatives import *
import scipy.sparse as sp
from .constants import *
from .pml import *
import numpy as np
from typing import *
from .utils import *

def eigen_2D_slice(
    L0: float, #L0 scaling parameter for distance units, usually 1e-6
    wvlen: float, # wvlen in units of L0
    xrange: np.array, #(xmin, xmax) in units of L0
    yrange: np.array, #(xmin, xmax) in units of L0
    eps_r,
    Npml, 
    polarization = 'TE'
):
    '''
        like curlcurlE, it is up to the user to use an algorithm to solve A for its eigenmodes
        this function just makes the operator
        
    '''
    omega = 2*np.pi*C0/(wvlen*L0);  # angular frequency in rad/sec

    ## we're just doing a 2D slice of a 3D system think of it like that
    ## generate PML parameters
    N2d = eps_r.shape
    L = np.array([np.diff(xrange)[0], np.diff(yrange)[0]]);
    dL2d = L/N2d
    
    #ETA0 is unitless so we don't need any scaling
    Sxfi, Sxbi, Syfi, Sybi, _,_ = S_create_3D(omega, [dL2d[0], dL2d[1], 1], [N2d[0], N2d[1], 1], [Npml[0], Npml[1],0], EPS0*L0, ETA0) #sp.identity(M);
    
    ## CREATE DERIVATIVES
    Dxf = Sxfi@createDws('x', 'f', dL2d, N2d); 
    Dxb = Sxbi@createDws('x', 'b', dL2d, N2d); 

    Dyf = Syfi@createDws('y', 'f', dL2d, N2d);
    Dyb = Sybi@createDws('y', 'b', dL2d, N2d); 
    
    eps_xx = bwdmean(eps_r, 'x')
    eps_yy = bwdmean(eps_r, 'y')
    eps_zz = eps_r
    
    M = np.prod(N2d)
    eps0 = EPS0*L0

    iTepz = sp.spdiags(1/(eps0*eps_zz.flatten(order = 'F')), 0, M,M);
    iTepx = sp.spdiags(1/(eps0*eps_xx.flatten(order = 'F')), 0, M,M);
    iTepy = sp.spdiags(1/(eps0*eps_yy.flatten(order = 'F')), 0, M,M);

    
    if(polarization == 'TE'):
        A = -(1/(MU0*L0))*(iTepz@(Dxb@Dxf + Dyb@Dyf))
    elif(self.polarization == 'TM'):
        A = -(1/(MU0*L0))*( Dxf@(iTepx)@Dxb + Dyf@(iTepy)@Dyb )
    return A


def eigen_slice_kz(
    L0: float, #L0 scaling parameter for distance units, usually 1e-6
    wvlen: float, # wvlen in units of L0
    xrange: np.array, #(xmin, xmax) in units of L0
    yrange: np.array, #(xmin, xmax) in units of L0
    eps_r,
    Npml, 
):
    '''
        eigensolver for a specific longitudinal wavevector kz (assuming waveguide axis is parallel to z)
        output modes are the hx and hy fields
    '''
    omega = 2*np.pi*C0/(wvlen*L0);  # angular frequency in rad/sec

    N2d = eps_r.shape
    L = np.array([np.diff(xrange)[0], np.diff(yrange)[0]]);
    dL2d = L/N2d
    eps0 = EPS0*L0;
    mu0 = MU0*L0
    M = np.prod(N2d)

    #ETA0 is unitless so we don't need any scaling
    Sxfi, Sxbi, Syfi, Sybi, _,_ = S_create_3D(omega, [dL2d[0], dL2d[1], 1], [N2d[0], N2d[1], 1], [Npml[0], Npml[1],0], EPS0*L0, ETA0) #sp.identity(M);
    
    ## CREATE DERIVATIVES
    Dxf = Sxfi@createDws('x', 'f', dL2d, N2d); 
    Dxb = Sxbi@createDws('x', 'b', dL2d, N2d); 

    Dyf = Syfi@createDws('y', 'f', dL2d, N2d);
    Dyb = Sybi@createDws('y', 'b', dL2d, N2d); 

    epxx= bwdmean(eps_r, 'x')
    epyy = bwdmean(eps_r,'y')

    Tez = sp.diags(eps0*eps_r.flatten(order = 'F'), 0, (M,M))
    Tey = sp.diags(eps0*epyy.flatten(order = 'F'), 0,  (M,M))
    Tex = sp.diags(eps0*epxx.flatten(order = 'F'), 0, (M,M))

    invTez = sp.diags(1/(eps0*eps_r.flatten(order = 'F')), 0,  (M,M))

    Dop1 = sp.bmat([[-Dyf], [Dxf]])
    Dop2 = sp.bmat([[-Dyb,Dxb]])
    Dop3 = sp.bmat([[Dxb], [Dyb]])
    Dop4 = sp.bmat([[Dxf,Dyf]])

    Tep = sp.block_diag((Tey, Tex))
    A =  Tep@(Dop1)@invTez@(Dop2) + Dop3@Dop4+ omega**2*mu0*Tep;
    
    return A


# def mode_filtering(
#    eigenmodes, 
#    eigenvals, 
#    structure_xbounds, 
#    structure_ybounds, 
#    L, 
#    Npml, 
#    pml_threshold = 1e-4
# ):

# # mode filtering only works for 2 dimensions...not recommended to do an eigensolve in 3D as it will default factorize the matrix
# # unless you're only interested in the largest eigenvalues

# #     %% assumes that xrange and yrange is [-, +] and is centered on 0
# #     % xlim: [x0, xf] x bounds of the STRUCTURE in PHYSICAL UNITS
# #     % ylim: [y0, yf] y bounds of the STRUCTURE in PHYSICAL UNITS (microns
# #     % or whatever)
# #     % eigenmodes should be a cell where each cell index is a field pattern
# #     % or mode pattern


    
# #     % mask KEYS
# #     % 2 = pml
# #     % 1 = structure
# #     % 0 = air;

#     N = size(eigenmodes{1});
#     Nx = N(1); Ny = N(2);
#     Nxc = round(Nx/2); Nyc = round(Ny/2);
#     x0 = structure_xbounds(1); 
#     xf = structure_xbounds(2); 
#     y0=structure_ybounds(1); 
#     yf = structure_ybounds(2);
    
#     # convert the physical bounds to grid bounds

#     Nx0 = Nxc+round((x0/L(1))*N(1))+1; Nxf = Nxc+floor((xf/L(1))*N(1));
#     Ny0 = Nyc+round((y0/L(2))*N(2))+1; Nyf = Nyc+floor((yf/L(2))*N(2));

#     #%% get PML bounds
#     x = np.arange(Nx)
#     y = np.arange(Ny) # x and y are node grids
#     [X,Y] = meshgrid(x,y);
#     #X = X.'; Y = Y.';
#     mask = np.zeros(N);
#     mask[(X<Npml(1) | X > Nx-Npml(1)) | ...
#             (Y<Npml(2) | Y> Ny - Npml(2))] = 2;
    
#     mask[Nx0:Nxf, Ny0:Nyf] = 1;
    
#     n = length(eigenmodes);
#     filtered_eigs = [];
#     filtered_modes = [];
#     c = 1;
    
#     #%% should we do an epsilon map of pml, air, and structure fields?
    
#     #%% execute the filters
#     for i in range(n):
        
#         structure_fields = eigenmodes{i}(mask == 1);
        
#         #get fields outside of structure
#         air_fields = eigenmodes{i}(mask == 0);
        
#         #get fields inside structure
#         PML_fields = eigenmodes{i}(mask == 2);
       
#         if(mean(mean(abs(PML_fields)))>pml_threshold):
#             disp('pml fields too large')
#             continue;   

#         if(mean(abs(structure_fields))> mean(abs(air_fields))):
#             filtered_eigs(c) =  eigenvals(i);
#             filtered_modes{c} = eigenmodes{i};
#             c = c+1;
#         else:
#             disp('too much field outside')
            
#     return filtered_modes, filtered_eigs, mask
