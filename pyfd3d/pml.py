import numpy as np
import scipy.sparse as sp

# EPSILON_0 = 8.85*10**-12
# MU_0 = 4*np.pi*10**-7
# ETA_0 = np.sqrt(MU_0/EPSILON_0)

def sig_w(l, dw, eta_0, m=3, lnR=-30):
    # helper for S()
    sig_max = -(m + 1) * lnR / (2 * eta_0 * dw)
    return sig_max * (l / dw)**m


def S(l, dw, omega, epsilon_0, eta_0):
    # helper for create_sfactor()
    return 1 - 1j * sig_w(l, dw, eta_0) / (omega * epsilon_0)


def create_sfactor(s, omega, dL, N, N_pml, epsilon_0, eta_0):
    # used to help construct the S matrices for the PML creation

    sfactor_vecay = np.ones(N, dtype=np.complex128)
    if N_pml < 1:
        return sfactor_vecay

    dw = N_pml * dL

    for i in range(N):
        if s == 'f':
            if i <= N_pml:
                sfactor_vecay[i] = S(dL * (N_pml - i + 0.5), dw, omega, epsilon_0, eta_0)
            elif i > N - N_pml:
                sfactor_vecay[i] = S(dL * (i - (N - N_pml) - 0.5), dw, omega, epsilon_0, eta_0)
        if s == 'b':
            if i <= N_pml:
                sfactor_vecay[i] = S(dL * (N_pml - i + 1), dw, omega, epsilon_0, eta_0)
            elif i > N - N_pml:
                sfactor_vecay[i] = S(dL * (i - (N - N_pml) - 1), dw, omega, epsilon_0, eta_0)
    return sfactor_vecay


def create_sc_pml(omega, dL, N, Npml, epsilon_0, eta_0):
    dx, dy, dz = dL
    Nx, Ny, Nz = N
    Nx_pml, Ny_pml, Nz_pml = Npml
    M = np.prod(N);
    
    sxf = create_sfactor('f', omega, dx, Nx, Nx_pml, epsilon_0, eta_0)
    syf = create_sfactor('f', omega, dy, Ny, Ny_pml, epsilon_0, eta_0)
    szf = create_sfactor('f', omega, dz, Nz, Nz_pml, epsilon_0, eta_0)
    
    sxb= create_sfactor('b', omega, dx, Nx, Nx_pml, epsilon_0, eta_0)
    syb= create_sfactor('b', omega, dy, Ny, Ny_pml, epsilon_0, eta_0)
    szb= create_sfactor('b', omega, dz, Nz, Nz_pml, epsilon_0, eta_0)
    
    #now we create the matrix (i.e. repeat sxf Ny times repeat Syf Nx times)
    [Sxf, Syf, Szf] = np.meshgrid(sxf, syf, szf, indexing = 'ij');
    [Sxb, Syb, Szb] = np.meshgrid(sxb, syb, szb, indexing = 'ij');
    
    return Sxf, Syf, Szf, Sxb, Syb, Szb

def S_create_3D(omega, dL, N, Npml, epsilon_0, eta_0):
    dx, dy, dz = dL
    Nx, Ny, Nz = N
    Nx_pml, Ny_pml, Nz_pml = Npml
    M = np.prod(N);
    
    Sxf, Syf, Szf, Sxb, Syb, Szb =  create_sc_pml(omega, dL, N, Npml, epsilon_0, eta_0);
    
    #Sxf(:) converts from n x n t0 n^2 x 1
    Sxfi=sp.spdiags(1/Sxf.flatten(order = 'F'),0,M,M);
    Sxbi=sp.spdiags(1/Sxb.flatten(order = 'F'),0,M,M);
    Syfi=sp.spdiags(1/Syf.flatten(order = 'F'),0,M,M);
    Sybi=sp.spdiags(1/Syb.flatten(order = 'F'),0,M,M);
    Szfi=sp.spdiags(1/Szf.flatten(order = 'F'),0,M,M);
    Szbi=sp.spdiags(1/Szb.flatten(order = 'F'),0,M,M);
    
    return Sxfi, Sxbi, Syfi, Sybi, Szfi, Szbi


def create_symmetrizer(Sxf, Syf, Szf, Sxb, Syb, Szb):
    
    sxf= Sxf.flatten(order = 'F')
    sxb = Sxb.flatten(order = 'F')
    
    syf = Syf.flatten(order = 'F')
    syb = Syb.flatten(order = 'F')
    
    szf = Szf.flatten(order = 'F')
    szb = Szb.flatten(order = 'F')
    
    numerator1 = np.sqrt((sxf*syb*szb));
    numerator2 = np.sqrt((sxb*syf*szb));
    numerator3 = np.sqrt((sxb*syb*szf));
    
    numerator = np.concatenate((numerator1, numerator2, numerator3), axis = 0);
    M =len(numerator);

    denominator = 1/numerator
    
    Pl = sp.spdiags(numerator, 0, M,M)
    Pr = sp.spdiags(denominator, 0, M,M);
    
    return Pl, Pr
                   
                  
    
    