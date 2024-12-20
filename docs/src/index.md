```@meta
CurrentModule = XGMRES
```

# Mixed precision preconditioning strategies for GMRES

This Julia package provides a mixed precision GMRES implementation with its set
of preconditioners for the solution of square general linear systems.

## Index

```@contents
Pages = ["xgmres.md","preconditioners.md","auxiliary.md"]
```

## How to use

## Disclaimers

The sole purpose of the XGMRES.jl GitHub repo is to provide the companion code
for the article “An investigation of mixed precision preconditioning strategies 
for GMRES” and provides the various scripts to reproduce the results therein.
Because this code WON'T BE MAINTAINED, it will not be made available through
the Julia package repository. Therefore, for people whose purpose is to use a 
reliable, standard, and performant Julia implementation of GMRES, we invite 
them to use more mature libraries such as 
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 

## Acknowledgements

We would like to acknowledge various repositories that helped us build this
work:
  - [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl)
  - [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl)
  - [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)
  - [SPAI-GMRES-IR](https://github.com/Noaman67khan/SPAI-GMRES-IR)
