# py-maxwell-fdfd
Solving Maxwell's equations via A python implementation of the 3D curl-curl E-field equations. This code contains additional work to engineer the eignspectrum for better convergence with iterative solvers (using the Beltrami-Laplace operator). You can control this in the main function through the input parameter $s = {0,-1,1}$

There is also a preconditioners to render the system matrix symmetric.

## Single Function Entry Point
The only function one really has to worry about is the one in fd3d.py This allows you to generate the matrix A and source vector b. Beyond that point, it is up to you to decide how to solve the linear system. There are some examples using scipy.sparse's qmr and bicg-stab in the notebooks but more likely than not, faster implementations exist elsewhere so it is important to have access to the underlying system matrix and right hand side. 

# important notes about implementation
1. Note that arrays are ordered using column-major (or Fortan) ordering whereas numpy is natively row-major or C ordering. You will see this in operations like reshape where I specify ordering (x.reshape(ordering = 'F')). It will also appear in meshgrid operations (use indexing = 'ij'). 

## Examples
1. Plane Wave (plane-wave test)

2. Dipole in Vacuum (vacuum.ipynb)

## Iterative Solvers
QMR and BICG-STAB are the first go-to solvers. Also for faster running, instead of using scipy's solvers, a number of other solvers:


## Future
Expect integration of this with ceviche for autograd
