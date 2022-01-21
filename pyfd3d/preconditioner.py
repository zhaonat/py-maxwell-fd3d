import numpy as np
import scipy.sparse as sp

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
                   