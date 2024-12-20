# XGMRES.jl

Mixed precision GMRES for the solution of square general linear systems written
in Julia.

A simple HTML documentation is available in `docs/build`.

## How to use

## Disclaimers

The sole purpose of the XGMRES.jl github repo is to provide the companion code
for the article "An investigation of mixed precision preconditioning strategies 
for GMRES" and provides the various scripts to reproduce the results therein.
Because this code WON'T BE MAINTAINED, it will not be made available through
the Julia package repository. Therefore, for people whose purpose is to use a 
reliable, standard, and performant Julia implementation of GMRES, we invite 
them to use more mature libraries such as 
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 




## Note to myself and collaborators

To launch remotely use

```bash
nohup julia --project scripts/dense_rand_gen.jl > logs.txt &
```
