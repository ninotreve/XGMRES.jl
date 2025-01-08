# XGMRES

The [`xgmres`](@ref) solver, combined with a given preconditioner, can make
use of up to six different floating point arithmetics. For
the sake of readability, we use the following set of consistent notations
to identify easily the precision at which a given variable is stored and an
operation is performed. These notations are consistent accross this 
documentation and the source code. You can find below a table listing the
different operations, their associated floating point datatype, and the 
subscript used for this floating point datatype.

| Operations | Floating point datatype | Subscript |
|:----------------|:-------|:------|
| Residual ``Ax_{i+1} - b`` | ``u_r`` | `r` | 
| Update ``x_{i+1} = x_{i} + d_{i}`` | ``u`` | no subscript |
| Preconditioner computation | ``u_x`` | `x` |
| Preconditioner application ``M\times v`` | ``u_m`` | `m` |
| Matrix application ``A\times v`` | ``u_a`` | `a` | 
| Rest of the GMRES operations | ``u_s`` | `s` |

For instance, if a variable is noted `varₓ`, this variable is stored in
precision ``u_x`` and is likely associated to the preconditioner computation.

The floating point arithmetic ``u`` is also refered to as the *working
precision*. This is the arithmetic at which the solution `x` of 
[`xgmres`](@ref) is delivered. We also use this notation when 
it is not pertinent to make a distinction between different floating point
arithmetics. Similarly, we do not add subscript to a variable in those cases.

Finally, in some parts of the code we refer to variables explicitly stored in 
`Float64` or `Float128` (through the 
[Quadmath](https://github.com/JuliaMath/Quadmath.jl) package). For those cases
we use the following subscripts.

| Floating point datatype | Subscript |
|:-------|:------|
|  `Float64` | `e` | 
|  `Floar128` | `t` |

Note that we would have liked subscripts `d` and `q`, respectively for `Float64`
(double precision) and `Float128` (quadruple precision), but Julia's unicode
does not have these options.

```@docs
XGMRES.xgmres
```

