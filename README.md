# py-maxwell-fd3d
Solving Maxwell's equations via A python implementation of the 3D curl-curl E-field equations. The primary purpose of this code is to expose the underlying techniques for generating finite differences in a relatively transparent way (so no classes or complicated interfaces). This code contains additional work to engineer the eignspectrum for better convergence with iterative solvers (using the Beltrami-Laplace operator). You can control this in the main function through the input parameter $s = {0,-1,1}$

There is also a preconditioners to render the system matrix symmetric.

## Single Function Entry Point
The only function one really has to worry about is the one in fd3d.py This allows you to generate the matrix A and source vector b. Beyond that point, it is up to you to decide how to solve the linear system. There are some examples using scipy.sparse's qmr and bicg-stab in the notebooks but more likely than not, faster implementations exist elsewhere so it is important to have access to the underlying system matrix and right hand side. 

# important notes about implementation
1. Note that arrays are ordered using column-major (or Fortan) ordering whereas numpy is natively row-major or C ordering. You will see this in operations like reshape where I specify ordering (x.reshape(ordering = 'F')). It will also appear in meshgrid operations (use indexing = 'ij'). 

## preconditioning-based approach for add-on functionality
Non-uniform grid can be implemented as a set of diagonal scaling preconditioners. This includes the sc-pml as well as smooth nonuniform-gridding

## Current Issues
1. So far, it appears that using scipy's iterative solvers, the case of the finite width photonic crystal slab has some issues with converging. scipy's lgmres and gcrotmk seems to work better, but are a lot slower than bicgstab or qmr

### Recommended Visualization in 3D: Plotly
see some of the examples below

## Examples

1. Dipole in Vacuum (vacuum.ipynb): a radiating dipole in vacuum with the domain truncated by a PML.
![Alt text](./img/vacuum_slices.png?raw=true "Title")

2. Plane Wave (plane-wave test)


3. 3D waveguide


4. Photonic Crystal Slab: lgmres
![Alt text](./img/phc_slab_slices.png?raw=true "Title")
![Alt text](./img/phc_eps_r_slices.png?raw=true "Title")


## Modal Sources
For now, note that my other set of codes, eigenwell has a number of mode solvers. You can solve a single 2d mode problem but this implicitly assumes kz = 0. There is another mode solver for $kz!=0$ as well there, but this is a bit more computationally expensive.

## Iterative Solvers
Unfortunately, given that iterative solvers don't have the same kind of robustness as factorization (i.e. convergence) combined with the fact that FDFD for Maxwell's equations are typically indefinite, iterative solving of equations is a bit more of an art than not. For different systems, solvers may converge reasonably or may not. 

For now, the solvers I've tried in scipy's sparse.linalg library are QMR and BICG-STAB and LGMRES. BICG-STAB and QMR are usually your go-to solvers but I've noticed some cases where LGMRES performs better.

External solvers include packages like petsc or suitesparse (but I'm still looking for good python interfaces for any external solvers).

## Direct Solvers
Direct solvers are robust but are incredibly memory inefficient, particulary for the curl-curl equations in 3D. If you want to experiment with solvers, try packages which support an LDL factorization for a complex symmetric matrix and also use block low rank compression (i.e. MUMPS).


## Future
Expect integration of this with ceviche for autograd
