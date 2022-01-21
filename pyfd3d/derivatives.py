import numpy as np
import scipy.sparse as sp
import numpy as np

def createDws(w, f, dL, N):
    '''
        s = 'x' or 'y': x derivative or y derivative
        f = 'b' or 'f'
        catches exceptions if s and f are misspecified
    '''
    M = np.prod(N);
  

    sign = 1 if f == 'f' else -1;
    dw = None; #just an initialization
    indices = np.reshape(np.arange(M), N, order = 'F');
    if(w == 'x'):
        ind_adj = np.roll(indices, -sign, axis = 0)
        dw = dL[0]
    elif(w == 'y'):
        ind_adj = np.roll(indices, -sign, axis = 1)
        dw = dL[1];
    elif(w == 'z'):
        ind_adj = np.roll(indices, -sign, axis = 2)
        dw = dL[-1]
        
    # we could use flatten here since the indices are already in 'F' order
    indices_flatten = np.reshape(indices, (M, ), order = 'F')
    indices_adj_flatten = np.reshape(ind_adj, (M, ), order = 'F')
    # on_inds = np.hstack((indices.flatten(), indices.flatten()))
    # off_inds = np.concatenate((indices.flatten(), ind_adj.flatten()), axis = 0);
    on_inds = np.hstack((indices_flatten, indices_flatten));
    off_inds = np.concatenate((indices_flatten, indices_adj_flatten), axis = 0);

    all_inds = np.concatenate((np.expand_dims(on_inds, axis =1 ), np.expand_dims(off_inds, axis = 1)), axis = 1)

    data = np.concatenate((-sign*np.ones((M)), sign*np.ones((M))), axis = 0)
    Dws = sp.csc_matrix((data, (all_inds[:,0], all_inds[:,1])), shape = (M,M));

    return (1/dw)*Dws;