import numpy as np
import scipy.sparse as sp

# function [grading] = tanh_grading(start_value, end_value, num_points)
    
#     % a tanh function appears to be smoother at the edges of the grid
#     % so I'm going to try this

#     center = min(start_value, end_value)+abs(end_value-start_value)/2;
#     xspace = linspace(start_value, end_value, num_points);
#     grading = tanh(xspace-center);
    
    
# end

# function grading = logarithmic_grading(h0, hf,N)
#     % alpha: grading factor
#     % N, number of steps to grade down to
#     grading = logspace(log10(h0), log10(hf), N);

# end

def non_uniform_scaling_operator(dx_scale, dy_scale):
    
#     %operators which perform the row-wise scaling
#     %xs: 1D array containing dx scalings (only for forward differences
    
    # create grid of x and y points
    [Xs, Ys] = meshgrid(dx_scale, dy_scale);
    %meshgrid isn't right for y
    M = numel(Xs);

    # we have to this kind of flip because the flattening
    # operation (:) doesn't retain row-major order
    Ys=Ys'; Xs = Xs';
    Fsy = spdiags(Ys(:),0,M,M);
    Fsx = spdiags(Xs(:),0,M,M);
    
    # might as well construct the conjugate grid.
    xc = (dx_scale+circshift(dx_scale,[0,1]))/2;
    yc = (dy_scale+circshift(dy_scale,[0,1]))/2;
    
    [Xc, Yc] = meshgrid(xc, yc);
    Xc = Xc';
    Yc = Yc';
    Fsy_conj = spdiags(Yc(:),0,M,M);
    Fsx_conj = spdiags(Xc(:),0,M,M);
    
    return Fsx, Fsy, Fsz Fsx_conj, Fsy_conj
    


def generate_nonuniform_scaling(Nft, drt):
    
    %Nft: 1st column is x, 2nd column is y
    %drt: list of discretizations...normalized by some reference
    % we can express drt as proportions of the largest discretization
    % available on the grid...but seems inefficient
    % advantage is we don't have to rewrite the pml sfactor

    Nx = sum(Nft(:,1));
    Ny = sum(Nft(:,2));
    dx_scale = ones(1,Nx);
    dy_scale = ones(1,Ny);
    
    num_regions = length(Nft(:,1));
    x0 = 1; y0 = 1;
    
#     % Here, we can assume that all odd indices are fixed regions
#     % even indices are transition regions
    
    for i = 1:2:num_regions
       dx_scale(x0:x0+Nft(i,1)-1) = drt(i,1);
       dy_scale(y0:y0+Nft(i,2)-1) = drt(i,2);
    

       if(i==num_regions) %no transition after last region
           x0 = x0+Nft(i,1);
           y0 = y0+Nft(i,2);
       else
           x0 = x0+Nft(i,1)+Nft(i+1,1);
           y0 = y0+Nft(i,2)+Nft(i+1,2);
       end
        
    end
    
    % do some sort of grading from region i to region i+1
    x0 = Nft(1,1); y0 = Nft(1,2);
    for i = 2:2:num_regions
        dx1 = drt(i-1,1); dx2 = drt(i+1,1);
        dy1 = drt(i-1,2); dy2 = drt(i+1,2);
        nxt = Nft(i,1); nyt = Nft(i,2);
        
        %need a function to grade smoothly from dr1 to dr2
        %equation to solve is there is some ration dr1/dr2 
        % need to multiply dr1 by constant nt times.

        grading_x = logspace(log10(dx1), log10(dx2), nxt+1);
        grading_y = logspace(log10(dy1), log10(dy2), nyt+1);

        dx_scale(x0:x0+nxt) = grading_x;
        dy_scale(y0:y0+nyt) = grading_y;
        x0 = x0+Nft(i,1)+Nft(i+1,1); 
        y0 = y0+Nft(i,2)+Nft(i+1,2);

    return