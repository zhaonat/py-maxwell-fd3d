# py-maxwell-fdfd
Solving Maxwell's equations via A python implementation of the 3D curl-curl E-field equations. This code contains additional work to engineer the eignspectrum for better convergence with iterative solvers (using the Beltrami-Laplace operator). You can control this in the main function through the input parameter $s = {0,-1,1}$

There is also a preconditioners to render the system matrix symmetric.

# important notes about implementation
1. Note that arrays are ordered using column-major (or Fortan) ordering whereas numpy is natively row-major or C ordering. You will see this in operations like reshape where I specify ordering (x.reshape(ordering = 'F')). It will also appear in meshgrid operations (use indexing = 'ij'). 

## Examples
1. Plane Wave

2. Dipole in Vacuum

## Iterative Solvers
QMR and BICG-STAB are the first go-to solvers. Also for faster running, instead of using scipy's solvers, a number of other solvers:


## Future
Expect integration of this with ceviche for autograd
