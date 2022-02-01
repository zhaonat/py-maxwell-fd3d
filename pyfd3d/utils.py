import numpy as np
import scipy.sparse as sp

def bwdmean(center_array, w):
    ## Input Parameters
    # center_array: 3D array of values defined at cell centers
    # w: 'x' or 'y' or 'z' direction in which average is taken

    ## Out Parameter
    # avg_array: 2D array of averaged values
    shift = 1
    if(w == 'y'):
        shift = 2 
    if(w == 'z'):
        shift = 3
    
    center_shifted = np.roll(center_array, shift); #doe sthis generalize easily into 3 Dimensions, CHECK!
    avg_array = (center_shifted + center_array) / 2;

    return avg_array