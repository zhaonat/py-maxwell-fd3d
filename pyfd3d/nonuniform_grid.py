import numpy as np
import scipy.sparse as sp

def tanh_grading(start_value, end_value, num_points):
    
#     % a tanh function appears to be smoother at the edges of the grid

    center = np.min(start_value, end_value)+np.abs(end_value-start_value)/2;
    xspace = np.linspace(start_value, end_value, num_points);
    grading = np.tanh(xspace-center);
    return grading
    

def logarithmic_grading(h0, hf,N):
    # alpha: grading factor
    # N, number of steps to grade down to
    grading = np.logspace(np.log10(h0), np.log10(hf), N);
    return grading


def non_uniform_scaling_operator(dx_scale, dy_scale, dz_scale):
    
#     %operators which perform the row-wise scaling
#     %xs: 1D array containing dx scalings (only for forward differences
    
    # create grid of x and y points
    [Xs, Ys, Zs] = np.meshgrid(dx_scale, dy_scale, dz_scale, indexing = 'ij');
    M = np.prod(Xs.shape);
                           
    Fsx = sp.spdiags(Xs.flatten(order = 'F'),0,M,M)
    Fsy = sp.spdiags(Ys.flatten(order = 'F'),0,M,M)
    Fsz = sp.spdiags(Zs.flatten(order = 'F'), 0, M,M)                    
    
    # might as well construct the conjugate grid.
    xc = (dx_scale + np.roll(dx_scale,[0,1]))/2;
    yc = (dy_scale + np.roll(dy_scale,[0,1]))/2;
    zc = (dz_scale + np.roll(dz_scale,[0,1]))/2;
    [Xc, Yc, Zc] = np.meshgrid(xc, yc, zc, indexing='ij')

    Fsy_conj = sp.spdiags(Yc.flatten(order = 'F'),0,M,M)
    Fsx_conj = sp.spdiags(Xc.flatten(order = 'F'),0,M,M)
    Fsz_conj = sp.spdiags(Zc.flatten(order = 'F'),0,M,M)
    
    return Fsx, Fsy, Fsz, Fsx_conj, Fsy_conj, Fsz_conj
    


def generate_nonuniform_scaling(
    Nft: np.array, 
    drt: np.array, 
):
    
    #Nft: 1st column is x, 2nd column is y
    #drt: list of discretizations...normalized by some reference
    # we can express drt as proportions of the largest discretization
    # available on the grid...but seems inefficient
    # advantage is we don't have to rewrite the pml sfactor

    Nx = np.sum(Nft[:,0])
    Ny = np.sum(Nft[:,1])
    Nz = np.sum(Nft[:,2])                          
    dx_scale = np.ones(Nx)
    dy_scale = np.ones(Ny)
    dz_scale = np.ones(Nx)
    num_regions = len(Nft)
    x0 = y0 = z0 = 0
    
#     % Here, we can assume that all odd indices are fixed regions
#     % even indices are transition regions
    
    for i in range(0, num_regions, 2): #= 1:2:num_regions
        dx_scale[x0:x0+Nft[i,0]] = drt[i,0]
        dy_scale[y0:y0+Nft[i,1]] = drt[i,1]
        dz_scale[z0:z0+Nft[i,2]] = drt[i,2]

        if(i==num_regions-1): #o transition after last region
            x0 = x0+Nft[i,0];
            y0 = y0+Nft[i,1];
            z0 = z0+Nft[i,2]
        else:
            x0 = x0+Nft[i,1]+Nft[i+1,0];
            y0 = y0+Nft[i,2]+Nft[i+1,1];
            z0 = z0+Nft[i,2]+Nft[i+1,2]

    
    # do some sort of grading from region i to region i+1
    x0 = Nft[0,0]
    y0 = Nft[0,1]
    z0 = Nft[0,2]   
                  
    for i in range(1, num_regions, 2): #= 2:2:num_regions
        dx1 = drt[i-1,0]; dx2 = drt[i+1,0];
        dy1 = drt[i-1,1]; dy2 = drt[i+1,1];
        dz1 = drt[i-1,2]; dz2 = drt[i+1,2];
                    
        nxt = Nft[i,0] 
        nyt = Nft[i,1]
        nzt = Nft[i,2]                     

        grading_x = np.logspace(np.log10(dx1), np.log10(dx2), nxt+1);
        grading_y = np.logspace(np.log10(dy1), np.log10(dy2), nyt+1);
        grading_z = np.logspace(np.log10(dz1), np.log10(dz2), nzt+1);
                               
        dx_scale[x0:x0+nxt+1] = grading_x;
        dy_scale[y0:y0+nyt+1] = grading_y;
        dz_scale[z0:z0+nzt+1] = grading_z;
                               
        x0 = x0+Nft[i,0]+Nft[i+1,0]; 
        y0 = y0+Nft[i,1]+Nft[i+1,1];
        z0 = z0+Nft[i,2]+Nft[i+1,2]                       

    return dx_scale, dy_scale, dz_scale