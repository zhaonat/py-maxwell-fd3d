import numpy as np
import scipy.sparse as sp
from .derivatives import *
from .pml import *
from typing import *
from .utils import *

def curlcurlE(
    L0: float, #L0 scaling parameter for distance units, usually 1e-6
    wvlen: float, # wvlen in units of L0
    xrange: np.array, #(xmin, xmax) in units of L0
    yrange: np.array, #(xmin, xmax) in units of L0
    zrange: np.array, #(xmin, xmax) in units of L0
    eps_r_tensor_dict: dict, 
    JCurrentVector: dict,
    Npml, 
    s = -1
):
    
    # normal SI parameters
    eps0 = 8.85*10**-12*L0;
    mu0 = 4*np.pi*10**-7*L0; 
    eta0 = np.sqrt(mu0/eps0);
    c0 = 1/np.sqrt(eps0*mu0);  # speed of light in 
    omega = 2*np.pi*c0/(wvlen);  # angular frequency in rad/sec
    
    eps_xx = eps_r_tensor_dict['eps_xx']
    eps_yy = eps_r_tensor_dict['eps_yy']
    eps_zz = eps_r_tensor_dict['eps_zz']
    
    ## edge smoothing of eps_xx, eps_yy, eps_zz
    eps_xx = bwdmean(eps_xx, 'x')
    eps_yy = bwdmean(eps_yy, 'y')
    eps_zz = bwdmean(eps_zz, 'z')
    
     
    N = eps_xx.shape;
    M = np.prod(N);
    
    L = np.array([np.diff(xrange)[0], np.diff(yrange)[0], np.diff(zrange)[0]]);
    dL = L/N
    
    Tepz = sp.spdiags(eps0*eps_zz.flatten(order = 'F'), 0, M,M);
    Tepx = sp.spdiags(eps0*eps_xx.flatten(order = 'F'), 0, M,M);
    Tepy = sp.spdiags(eps0*eps_yy.flatten(order = 'F'), 0, M,M);

    iTepz = sp.spdiags(1/(eps0*eps_zz.flatten(order = 'F')), 0, M,M);
    iTepx = sp.spdiags(1/(eps0*eps_xx.flatten(order = 'F')), 0, M,M);
    iTepy = sp.spdiags(1/(eps0*eps_yy.flatten(order = 'F')), 0, M,M);
    
    iTepsSuper = sp.block_diag((iTepx, iTepy, iTepz));
    TepsSuper = sp.block_diag((Tepx, Tepy, Tepz));

    iTmuSuper = (1/mu0)*sp.identity(3*M)
    TmuSuper = (mu0)*sp.identity(3*M)
    
    ## generate PML parameters
    # Sxf = sp.identity(3*M);
    Sxfi, Sxbi, Syfi, Sybi, Szfi, Szbi = S_create_3D(omega, dL, N, Npml, eps0, eta0) #sp.identity(M);
    
    ## CREATE DERIVATIVES
    Dxf = Sxfi@createDws('x', 'f', dL, N); 
    Dxb = Sxbi@createDws('x', 'b', dL, N); 

    Dyf = Syfi@createDws('y', 'f', dL, N);
    Dyb = Sybi@createDws('y', 'b', dL, N); 
    
    Dzf = Szfi@createDws('z', 'f', dL, N); 
    Dzb = Szbi@createDws('z', 'b', dL, N); 
    
    
    #curlE and curlH
    Ce = sp.bmat([[None, -Dzf, Dyf], 
                  [Dzf, None, -Dxf], 
                  [-Dyf, Dxf, None]])
    Ch = sp.bmat([[None, -Dzb, Dyb], 
                  [Dzb, None, -Dxb], 
                  [-Dyb, Dxb, None]])
    
    ##graddiv, aka beltrami-laplace from Wonseok's paper
    gd00 = Dxf@iTepx@Dxb@Tepx
    gd01 = Dxf@iTepx@(Dyb@Tepy)
    gd02 = Dxf@iTepx@(Dzb@Tepz)
    
    gd10 = Dyf@iTepy@(Dxb@Tepx)
    gd11 = Dyf@iTepy@(Dyb@Tepy)
    gd12 = Dyf@iTepy@(Dzb@Tepz)
    
    gd20 = Dzf@iTepz@(Dxb@Tepx)
    gd21 = Dzf@iTepz@(Dyb@Tepy)
    gd22 = Dzf@iTepz@(Dzb@Tepz)

    GradDiv = sp.bmat([[gd00, gd01, gd02],
                       [gd10, gd11, gd12],
                       [gd20, gd21, gd22]]);
    
    WAccelScal = sp.identity(3*M)@iTmuSuper;
    A = Ch@iTmuSuper@Ce + s*WAccelScal@GradDiv - omega**2*TepsSuper;
    
    ## source setup
    Jx = JCurrentVector['Jx'].flatten(order = 'F')
    Jy = JCurrentVector['Jy'].flatten(order = 'F')
    Jz = JCurrentVector['Jz'].flatten(order = 'F')
    
    J = np.concatenate((Jx,Jy,Jz), axis = 0);
    print(J.shape)
    
    b = -1j*omega*J; 
    JCorrection = (1j/omega) * (s*GradDiv@WAccelScal)@iTepsSuper@J;
    b = b+JCorrection;
    
    return A,b, Ch # last arg let's you recover H fields

    
