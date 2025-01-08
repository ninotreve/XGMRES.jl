# XGMRES.jl

This Julia package provides a mixed precision GMRES implementation with its set
of preconditioners for the solution of square general linear systems. It is the
companion code of the academic article "Mixed precision preconditioning 
strategies for GMRES".

A small HTML documentation is available in `docs/build`.

## How to use

Using the function `xgmres` can be as simple as writing
```julia
A = rand(100,100)
b = rand(100)
x, _ = xgmres(A,b)
```
To go further, other parameters are fine-tuneable, including the 
precisions at which the operations are performed. The complete list of the 
parameters is provided in the function documentation.

The `scripts/dense.jl` and `scripts/sparse.jl` files in this repo are good
examples of applications of the function. The first script runs `xgmres` on a
random dense synthetic problem generated with target condition numbers for $A$ 
and for the preconditioner. The second script loads a SuiteSparse matrix and 
runs `xgmres` on this problem with one of our built-in preconditioners. These
files can
also be used to reproduce or check the results presented in (most of) the plots
of the article. 

These scripts can be conveniently run in the background with 
the Bash commands
```bash
sh ./scripts/scripts.sh dense.jl
sh ./scripts/scripts.sh sparse.jl
```
or launch directly with Julia from the root of the project
```bash
julia --project=. scripts/sparse.jl
julia --project=. scripts/sparse.jl
```
## Disclaimers

The sole purpose of the XGMRES.jl GitHub repo is to provide the 
Julia code used to generate the numerical results of the academic paper
"Mixed precision preconditioning strategies for GMRES". This code WON'T BE 
MAINTAINED and won't be made available through the Julia package repository.
Hence, we advise people looking for a reliable, standard, and performant Julia 
implementation of GMRES to use the more mature 
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) library.

## Acknowledgements

We would like to acknowledge various repositories that helped us build this
work:
  - [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl)
  - [SPAI-GMRES-IR](https://github.com/Noaman67khan/SPAI-GMRES-IR)
  - [BFloat16s](https://github.com/JuliaMath/BFloat16s.jl)
  - [Quadmath.jl](https://github.com/JuliaMath/Quadmath.jl)
  - [MatrixMarket.jl](https://github.com/JuliaSparse/MatrixMarket.jl)
  - [SuiteSparseMatrixCollection.jl](https://github.com/JuliaSmoothOptimizers/SuiteSparseMatrixCollection.jl)
