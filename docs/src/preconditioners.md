# Preconditioners

The package wraps and implements various preconditioners. The complete list is
given below:

| Preconditioners | Method | Dense | Sparse |
|:----------------|:-------|:------|:-------|
| Identity | [`create_precond_I`](@ref) | ✓ | ✓ |
| LU | [`create_precond_lu`](@ref) | ✓ |  |
| ILU | [`create_precond_ilu`](@ref) |  | ✓ |
| AMG | [`create_precond_amg`](@ref) |  | ✓ |
| Polynomial | [`create_precond_poly`](@ref) | ✓ | ✓ |
| SPAI | [`create_precond_spai`](@ref) |  | ✓ |
| Random[^1] | [`create_precond_rand`](@ref) | ✓  |  |

[^1]: Only useable when the problems are generated with 
      [`gen_mat_with_prec`](@ref).

To have a consistent interface to be used inside [`xgmres`](@ref), the 
preconditioners are wrapped in the following structure. 

```@docs
XGMRES.Preconditioner
```

## List of available preconditioners

```@docs
XGMRES.create_precond_I
XGMRES.create_precond_lu
XGMRES.create_precond_ilu
XGMRES.create_precond_amg
XGMRES.create_precond_poly
XGMRES.create_precond_spai
XGMRES.create_precond_rand
```

## Sparse approximate inverse

```@docs
XGMRES.spai
```

## Polynomial preconditioner

```@docs
XGMRES.Poly
XGMRES.polynomial
Base.:*
```

