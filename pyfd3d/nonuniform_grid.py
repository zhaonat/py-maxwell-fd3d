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
#.     these operators have to be applied to the individual dxf, dyf, operators...no global operators allowed
    
    # create grid of x and y points
    [Xs, Ys, Zs] = np.meshgrid(dx_scale, dy_scale, dz_scale, indexing = 'ij');
    M = np.prod(Xs.shape);
                           
    Fsxi = sp.spdiags(1/Xs.flatten(order = 'F'), 0, M,M)
    Fsyi = sp.spdiags(1/Ys.flatten(order = 'F'), 0, M,M)
    Fszi = sp.spdiags(1/Zs.flatten(order = 'F'), 0, M,M)                    
    
    # might as well construct the conjugate grid.
    xc = (dx_scale + np.roll(dx_scale, -1))/2; #note differencesroll vs matlab's circshift
    yc = (dy_scale + np.roll(dy_scale, -1))/2;
    zc = (dz_scale + np.roll(dz_scale, -1))/2;
    [Xc, Yc, Zc] = np.meshgrid(xc, yc, zc, indexing='ij')

    Fsyi_conj = sp.spdiags(1/Yc.flatten(order = 'F'),0,M,M)
    Fsxi_conj = sp.spdiags(1/Xc.flatten(order = 'F'),0,M,M)
    Fszi_conj = sp.spdiags(1/Zc.flatten(order = 'F'),0,M,M)
    
    # return like this...
    return Fsxi, Fsyi, Fszi, Fsxi_conj, Fsyi_conj, Fszi_conj
    


# this function must be flexible enough to accept a nonuniform grid in subsets of cartesian axes
def generate_nonuniform_scaling(
    Nft: np.array, 
    drt: np.array, 
) -> list:
    
    #Nft: 1st column is x, 2nd column is y
    #drt: list of discretizations...normalized by some reference
    # we can express drt as proportions of the largest discretization
    # available on the grid...but seems inefficient
    # advantage is we don't have to rewrite the pml sfactor
    _, dimension = Nft.shape; #dimension is the number of coordinates we're dealing with
    dr_scalings = []
    num_regions = len(Nft)

    for i in range(dimension):
        Ni = np.sum(Nft[:,i])
        di_scale = np.ones(Ni)
        i0 = 0;
        for j in range(0, num_regions, 2):
            di_scale[i0:i0+Nft[j,i]] = drt[j,i];
            if(j== num_regions-1):
                i0 = i0+Nft[j,i];
            else:
                i0 = i0+Nft[j,i]+Nft[j+1,i];
        i0 = Nft[0,0] 
        for j in range(1, num_regions, 2): # these are the transition regions
            di1 = drt[j-1,i]; di2 = drt[j+1,i];
            nit = Nft[j,i] 
            grading_i = np.logspace(np.log10(di1), np.log10(di2), nit+1);
            di_scale[i0:i0+nit+1] = grading_i;
            i0 = i0+Nft[j,i]+Nft[j+1,i]; 
        dr_scalings.append(di_scale);
    return dr_scalings #list
        
#     Nx = np.sum(Nft[:,0])
#     Ny = np.sum(Nft[:,1])
#     Nz = np.sum(Nft[:,2])                          
#     dx_scale = np.ones(Nx)
#     dy_scale = np.ones(Ny)
#     dz_scale = np.ones(Nx)
#     num_regions = len(Nft)
#     x0 = y0 = z0 = 0
    
# #     % Here, we can assume that all odd indices are fixed regions
# #     % even indices are transition regions
    
#     for i in range(0, num_regions, 2): #coarse regions
#         dx_scale[x0:x0+Nft[i,0]] = drt[i,0]
#         dy_scale[y0:y0+Nft[i,1]] = drt[i,1]
#         dz_scale[z0:z0+Nft[i,2]] = drt[i,2]

#         if(i==num_regions-1): #o transition after last region
#             x0 = x0+Nft[i,0];
#             y0 = y0+Nft[i,1];
#             z0 = z0+Nft[i,2]
#         else:
#             x0 = x0+Nft[i,1]+Nft[i+1,0];
#             y0 = y0+Nft[i,1]+Nft[i+1,1];
#             z0 = z0+Nft[i,2]+Nft[i+1,2]

    
#     # do some sort of grading from region i to region i+1
#     x0 = Nft[0,0] #x0 represents end point
#     y0 = Nft[0,1]
#     z0 = Nft[0,2]   
                  
#     for i in range(1, num_regions, 2): # these are the transition regions
#         dx1 = drt[i-1,0]; dx2 = drt[i+1,0];
#         dy1 = drt[i-1,1]; dy2 = drt[i+1,1];
#         dz1 = drt[i-1,2]; dz2 = drt[i+1,2];
                    
#         nxt = Nft[i,0] 
#         nyt = Nft[i,1]
#         nzt = Nft[i,2]   
#         print(nxt, nyt, nzt, dx1, dx2, dy1, dy2)

#         grading_x = np.logspace(np.log10(dx1), np.log10(dx2), nxt+1);
#         grading_y = np.logspace(np.log10(dy1), np.log10(dy2), nyt+1);
#         grading_z = np.logspace(np.log10(dz1), np.log10(dz2), nzt+1);
        
#         dx_scale[x0:x0+nxt+1] = grading_x;
#         dy_scale[y0:y0+nyt+1] = grading_y;
#         dz_scale[z0:z0+nzt+1] = grading_z;
                               
#         x0 = x0+Nft[i,0]+Nft[i+1,0]; 
#         y0 = y0+Nft[i,1]+Nft[i+1,1];
#         z0 = z0+Nft[i,2]+Nft[i+1,2]                       

#     return dx_scale, dy_scale, dz_scale