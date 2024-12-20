# File: XGMRES.jl
# Author: Bastien Vieublé
# Email: bastien.vieuble@irit.fr
# Github: https://github.com/bvieuble/XGMRES.jl
# Description: Module containing a mixed precision preconditioned GMRES using 
# arbitrary preconditioner and precisions. The code is not particularly
# optimized and is intended to be used only for numerical purpose. 

module XGMRES

import Base: *

using LinearAlgebra, SparseArrays, Random, Quadmath, Printf, IncompleteLU,
    AlgebraicMultigrid, BFloat16s, ILUZero

export xgmres, Preconditioner, create_precond_I, gen_mat, gen_mat_with_prec,
    create_precond_rand, xconvert, create_precond_ilu, create_precond_lu,
    create_precond_amg, create_precond_spai, create_precond_poly, 
    create_precond_ilu0, create_precond_rand_inv

include("aux.jl")
include("rand_mat.jl")
include("spai.jl")
include("polynomial.jl")
include("preconditioner.jl")

"""
```julia
x::Vector{u}, stats::Dict = xgmres(
    A         ::AbstractMatrix{TA},                  
    b         ::AbstractVector{TB};                  
    precond   ::Preconditioner            = nothing, 
    kind      ::String                    = "left",  
    xexact    ::Union{Vector{TX},Nothing} = nothing, 
    maxrestrt ::Union{Integer,   Nothing} = nothing, 
    m         ::Union{Integer,   Nothing} = nothing, 
    τ         ::Union{Float64,   Nothing} = nothing, 
    bstop     ::Union{Float64,   Nothing} = nothing, 
    fstop     ::Union{Float64,   Nothing} = nothing, 
    stop      ::Union{Float64,   Nothing} = nothing, 
    verbose   ::Bool                      = true,    
    do_stats  ::Bool                      = false,   
    do_κ      ::Bool                      = false,   
    u         ::DataType                  = Float64, 
    uᵣ        ::DataType                  = Float64, 
    uₛ        ::DataType                  = Float64, 
    maxmem    ::Float64                   = 3*1e9    
) where {TA<:AbstractFloat,TB<:AbstractFloat,TX<:AbstractFloat}
```
Solve the linear system ``Ax=b`` with mixed precision restarted preconditioned 
GMRES, where ``A\\in\\mathbb{R}^{n\\times n}`` and ``x,b \\in\\mathbb{R}^n``.
The function requires three precision parameters and repeats the three 
following steps until convergence:
1. Compute the residual ``r_i = b-Ax_i`` in precision ``u_r``
2. Solve ``A d_i = r_i/||r_i||_{\\infty}`` with preconditioned GMRES run in 
   precision ``u_s``
3. Update the solution ``x_{i+1} = x_i + ||r_i||_{\\infty}\\times d_i`` in 
   precision ``u``

The precisions ``u_r`` and ``u``  concern, respectively, the precisions at 
which the residual (step 1) and the update of the solution (step 3) are 
computed. The precision ``u_s`` is the precision at which all the operations of
the inner GMRES, except the preconditioned matrix-vector product, are computed
(step 2). The preconditioned matrix-vector product should be provided to the 
function via the `precond` parameter.

!!! note "Preconditioned matrix-vector product"
    The preconditioned matrix-vector product can use its own set of precisions 
    and does not have to be implemented in precision ``uₛ``. One should use 
    the `Preconditioner` structure provided by the package to implement it or 
    use provided implementations by the package. 

The algorithm is composed of two imbricated loops: the outer restart loop and 
the inner GMRES iterations.
The restart loop is stopped if:
  * We reached the maximum number of restart iterations `maxrestrt`.
  * We observe no significant error reduction over three consecutives 
    iterations. This is quantified
    by the parameter `stop` which defines the maximum acceptable ratio between 
    the forward errors of two consecutive iterations. 
  * We reached the expected accuracies on the forward and backward errors 
    defined by the parameters `fstop` and `bstop`.
If the parameter `xexact` is not provided, the forward error is not computed, 
and the above stopping criterion are solely based on the computed backward 
error. The backward and forward error of the linear system are computed, 
respectively, as follows:

\$\\frac{||Ax_i-b||_{\\infty}}{||A||_{\\infty}||x_i||_{\\infty} + 
||b||_{\\infty}},\$

\$\\frac{||x_i - x||_{\\infty}}{||x||_{\\infty}}.\$

The inner GMRES loop is stopped if:
  * We reached the maximum number of GMRES iterations `m`.
  * The backward error of the preconditioned system ``Ad_i=r_i`` falls below a 
    given tolerance defined by the paramter `τ`.

The backward error of the preconditioned system is computed as follows:

\$\\frac{||M^{-1}(Ad_{i,j} - r_i/||r_i||_{\\infty})||_{\\infty}}{||M^{-1}r_i/
||r_i||_{\\infty}||_{\\infty}} \\qquad \\text{(left-preconditioned)}\$ 

\$\\frac{||Ad_{i,j} - r_i/||r_i||_{\\infty})||_{\\infty}}{||r_i/
||r_i||_{\\infty}||_{\\infty}} \\qquad \\text{(right-preconditioned)}\$ 

!!! note "Right-preconditioned backward error"
    The backward error of the right-preconditioned system is the backward error
    of the original system ``A d_i = r_i/||r_i||_{\\infty}``.

Through the parameter `do_stats`, it can be asked to the function to compute 
the backward errors of the original system \$Ax=b\$ for each iteration of GMRES. 
The forward errors are also computed if the exact solution of the system is 
provided through the parameter `xexact`. The condition numbers of the 
preconditioner and the preconditioned matrix can also be computed through 
another parameter `do_κ`. It lets the possibility to compute the `do_stats` 
advanced errors on systems where the condition number is too expensive or 
impossible to compute.

!!! warning "Stats mode"
    The stat mode increases substantially the resource comsumption and is only 
    relevant for numerical study of the algorithm. It is not intended to be 
    used in practice.

!!!TODO: DEFINE THE STAT STRUCTURE!!!

**Input arguments**

  - `A`: A square general matrix ``A\\in\\mathbb{R}^{n\\times n}``
  - `b`: A right-hand side ``b\\in\\mathbb{R}^n``

**Keyword arguments**

  - `precond`: A [`Preconditioner`](@ref) object embedding the linear operators 
    used for preconditioning
  - `kind`: Set the kind of preconditioning ("left", "right", or "flexible")
  - `xexact`: Exact solution of the system. If provided the forward error is 
    computed in stat mode.
  - `maxrestrt`: Maximum number of restart iterations.
  - `m`: Maxmum number of inner GMRES iterations.
  - `τ`: Stopping criterion for the inner iterations of GMRES based on the 
    backward error of the correction system ``A d_i = r_i/||r_i||_{\\infty}``.
  - `bstop`: Stopping criterion for the outer restart iterations based on the 
    backward error of the original system ``A x_i = b``.
  - `fstop`: Stopping criterion for the outer restart iterations based on the 
    forward error of the original system.
  - `stop`: Stopping criterion for the outer restart iterations based on the 
    based on the non-improvement of the forward error over 3 iterations. If the
    forward error is not improved by a ratio higher than `stop` for three 
    consecutive restart iterations, the algorithm is stopped.
  - `verbose`: Information and convergence logs are displayed at each iteration.
  - `do_stats`: Activate the stats mode and get many more logs.
  - `do_κ`: Compute the condition numbers of the preconditioner ``M`` and the
    preconditioned matrix ``AM`` or ``MA`` (depending on the kind of 
    preconditioning used).
  - `u`: Precision at which the update ``x_{i+1} = x_{i} + d_{i}`` is computed.
  - `uᵣ`: Precision at which the residual ``Ax_{i+1} - b`` is computed.
  - `uₛ`: Precision at which the operations in the inner loop of GMRES are 
    computed.
  - `maxmem`: Use to set `m` if `m` is not provided in argument. It chooses `m`
    such that the memory used to store the Krylov bases is lower than `maxmem`.
"""
function xgmres(
    A::AbstractMatrix{TA},
    b::AbstractVector{TB};
    precond::Preconditioner=nothing,
    kind::String="left",
    xexact::Union{Vector{TX},Nothing}=nothing,
    maxrestrt::Union{Integer,Nothing}=nothing,
    m::Union{Integer,Nothing}=nothing,
    τ::Union{Float64,Nothing}=nothing,
    bstop::Union{Float64,Nothing}=nothing,
    fstop::Union{Float64,Nothing}=nothing,
    stop::Union{Float64,Nothing}=nothing,
    verbose::Bool=true,
    do_stats::Bool=false,
    do_κ::Bool=false,
    u::DataType=Float64,
    uᵣ::DataType=Float64,
    uₛ::DataType=Float64,
    maxmem::Float64=3 * 1e9
) where {TA<:AbstractFloat,TB<:AbstractFloat,TX<:AbstractFloat}

    if (maxrestrt === nothing)
        max_it = 1
    else
        max_it = maxrestrt
    end

    if (τ === nothing)
        τ = eps(uₛ) * 2
    end

    if (bstop === nothing)
        bstop = eps(u) * 2
    end

    if (fstop === nothing)
        fstop = eps(u) * 2
    end

    if (m === nothing)
        if uₛ == Float16 || uₛ == BFloat16
            bytes = 2
        elseif uₛ == Float32
            bytes = 4
        elseif uₛ == Float64
            bytes = 8
        elseif uₛ == Float128
            bytes = 16
        else
            error("Unknown arithmetic precision for uₛ.")
        end

        n = size(b, 1)
        m = floor(Int64, (-(n * bytes + bytes) + sqrt((n * bytes + bytes)^2 +
                                                      4 * bytes * maxmem)) / (2 * bytes))
        m = min(m, n)
    end

    if (precond === nothing)
        precond = create_precond_I(A, uₛ)
    end

    if (eltype(b) != uᵣ)
        bᵣ = xconvert(uᵣ, b)
    else
        bᵣ = b
    end

    if (eltype(A) != uᵣ)
        Aᵣ = xconvert(uᵣ, A)
    else
        Aᵣ = A
    end

    n = size(b, 1)
    Vₛ = zeros(uₛ, n, m + 1)
    if kind == "flexible"
        Zₛ = zeros(uₛ, n, m + 1)
    end
    Hₛ = zeros(uₛ, m + 1, m)
    Gₛ = Array{Tuple{LinearAlgebra.Givens{uₛ},uₛ},1}(undef, n)
    e₁ = zeros(uₛ, n + 1)
    e₁[1] = 1.0
    nit = 0
    tmp = 0
    err = 1.0
    bkw = Float64[]
    fwd = Float64[]
    bkwall = Float64[]
    fwdall = Float64[]
    precbkw = Float64[]
    bnrmInf = xconvert(uᵣ, norm(b, Inf))
    AnrmInf = xconvert(uᵣ, opnorm(A, Inf))
    stats = Dict("cvg" => 0,
        "gmresits" => Int32[],
        "precbkw" => Float64[],
        "err" => Float64[],
        "m" => m,
    )
    push!(stats["gmresits"], 0)

    bprec = xconvert(u, precond.Mx(b, do_stats)[1])
    x = bprec

    if do_κ
        # Compute the norms of the preconditioned right-hand side and matrix 
        # for the computation of the preconditioned backward error. Compute the 
        # conditiion number of M and AM/MA.
        if kind == "left"
            MA = precond.MA()
            M = precond.M()
            bprecnrmInf = norm(bprec, Inf)
            AprecnrmInf = opnorm(MA, Inf)
            try
                stats["K(MA)"] = cond(MA, Inf)
            catch e
                if isa(e, SingularException)
                    stats["K(MA)"] = Inf
                else
                    print(e, "\n")
                    error("ERROR: Error while computing κ(MA).\n")
                end
            end
            try
                stats["K(M)"] = cond(M, Inf)
            catch e
                if isa(e, SingularException)
                    stats["K(M)"] = Inf
                else
                    print(e, "\n")
                    error("ERROR: Error while computing κ(M).\n")
                end
            end
        else
            AM = precond.AM()
            M = precond.M()
            bprecnrmInf = bnrmInf
            AprecnrmInf = AnrmInf
            try
                stats["K(AM)"] = cond(AM, Inf)
            catch e
                if isa(e, SingularException)
                    stats["K(AM)"] = Inf
                else
                    print(e, "\n")
                    error("ERROR: Error while computing κ(AM).\n")
                end
            end
            try
                stats["K(M)"] = cond(M, Inf)
            catch e
                if isa(e, SingularException)
                    stats["K(M)"] = Inf
                else
                    print(e, "\n")
                    error("ERROR: Error while computing κ(M).\n")
                end
            end
        end
    else
        # For convenience, if the do_κ is not activated we still initialize the 
        # variable AprecnrmInf and bprecnrmInf for the preconditioned backward 
        # error to be computed and displayed. However, for the left-preconditioned
        # case, the preconditioned backward error displayed won't be good.
        bprecnrmInf = bnrmInf
        AprecnrmInf = AnrmInf
    end

    iter = 0
    while true
        iter += 1

        xᵣ = xconvert(uᵣ, x)
        rᵣ = bᵣ - Aᵣ * xᵣ
        rnrmInf = norm(rᵣ, Inf)
        srᵣ = rᵣ / rnrmInf # Scaling the residual for more stability

        if (xexact !== nothing)
            xnrmInf = xconvert(uᵣ, norm(xᵣ, Inf))
            push!(bkw,
                xconvert(Float64, rnrmInf / (AnrmInf * xnrmInf + bnrmInf)))
            push!(fwd,
                xconvert(Float64, norm(xexact - xconvert(eltype(xexact), x),
                    Inf) / norm(xexact, Inf)))
            if (verbose)
                @printf("restrt: %2d --- bkw = %.5e --- fwd = %.5e ", iter,
                    bkw[end],
                    fwd[end])
                @printf("--- gmresits = %d\n", nit - tmp)
                tmp = nit
            end
            if (bkw[end] < bstop && fwd[end] < fstop)
                stats["cvg"] = 1
                break
            end
        else
            xnrmInf = xconvert(uᵣ, norm(xᵣ, Inf))
            push!(bkw, rnrmInf / (AnrmInf * xnrmInf + bnrmInf))
            if (verbose)
                @printf("restrt: %2d --- bkw = %.5e --- gmresits = %d\n",
                    iter, bkw[end], nit - tmp)
                tmp = nit
            end
            if (bkw[end] <= bstop)
                stats["cvg"] = 1
                break
            end
        end

        if iter > max_it
            break
        end

        if stop !== nothing && iter > 3
            cond1 = (fwd[end] / fwd[end-1] > stop
                     &&
                     fwd[end-1] / fwd[end-2] > stop)
            cond2 = (bkw[end] / bkw[end-1] > stop
                     &&
                     bkw[end-1] / bkw[end-2] > stop)
            if cond1 && cond2
                stats["cvg"] = 0
                break
            end
            cond3 = Base.isnan(fwd[end]) || Base.isnan(bkw[end])
            if cond3
                stats["cvg"] = 0
                break
            end
        end

        if kind == "left"
            rₛ = xconvert(uₛ, precond.Mx(srᵣ, do_stats)[1])
        elseif kind == "right" || kind == "flexible"
            rₛ = xconvert(uₛ, srᵣ)
        else
            error("GMRES kind unknown.")
        end

        rnrm_prec = norm(rₛ, 2)
        Vₛ[:, 1] = rₛ ./ rnrm_prec
        sₛ = rnrm_prec * e₁

        for i = 1:m
            nit = nit + 1
            if kind == "left"
                w, stat = precond.MAx(Vₛ[:, i], do_stats)
                wₛ = xconvert(uₛ, w)
            elseif kind == "right"
                w, _, stat = precond.AMx(Vₛ[:, i], do_stats)
                wₛ = xconvert(uₛ, w)
            elseif kind == "flexible"
                w, z, stat = precond.AMx(Vₛ[:, i], do_stats)
                Zₛ[:, i] = xconvert(uₛ, z)
                wₛ = xconvert(uₛ, w)
            else
                error("GMRES kind unknown.")
            end

            collect_stats(stats, stat)

            for k = 1:i
                Hₛ[k, i] = wₛ' * Vₛ[:, k]
                wₛ = wₛ - Hₛ[k, i] * Vₛ[:, k]
            end
            Hₛ[i+1, i] = norm(wₛ, 2)
            Vₛ[:, i+1] = wₛ / Hₛ[i+1, i]
            for k = 1:i-1
                Hₛ[:, i] = (Gₛ[k])[1] * Hₛ[:, i]
            end
            Gₛ[i] = givens(Hₛ[:, i], i, i + 1)
            sₛ[:] = (Gₛ[i])[1] * sₛ
            Hₛ[i, i] = (Gₛ[i])[2]
            Hₛ[i+1, i] = 0.0
            err = abs(sₛ[i+1]) / rnrm_prec
            push!(stats["err"], xconvert(Float64, err))

            if do_stats
                # Computation of the errors of the original system
                yₛ = Hₛ[1:i, 1:i] \ sₛ[1:i]
                if kind == "left"
                    addvec = Vₛ[:, 1:i] * yₛ
                elseif kind == "right"
                    addvec = precond.Mx(Vₛ[:, 1:i] * yₛ, do_stats)[1]
                elseif kind == "flexible"
                    addvec = Zₛ[:, 1:i] * yₛ
                else
                    error("GMRES kind unknown.")
                end
                if xexact !== nothing
                    xnrmInf = xconvert(uᵣ, norm(xexact, Inf))
                else
                    xnrmInf = xconvert(uᵣ, norm(xᵣ, Inf))
                end
                x_stats = x + xconvert(u, rnrmInf) * xconvert(u, addvec)
                push!(bkwall,
                    xconvert(Float64, norm(Aᵣ * xconvert(uᵣ, x_stats)
                                           -
                                           bᵣ, Inf) / (AnrmInf * xnrmInf + bnrmInf)))
                if xexact !== nothing
                    push!(fwdall,
                        xconvert(Float64, norm(xexact -
                                               xconvert(eltype(xexact), x_stats), Inf) /
                                          norm(xexact, Inf)))
                end

                # Compputation of the backward error of the preconditioned 
                # system.
                prec_bkw = abs(sₛ[i+1]) * xconvert(uₛ, rnrmInf) /
                           xconvert(uₛ, (AprecnrmInf * xnrmInf + bprecnrmInf))
                push!(precbkw, xconvert(Float64, prec_bkw))
            end

            if (verbose && do_stats && xexact !== nothing)
                @printf("       ---> it: %2d --- err = %.5e --- prec bkw = \
                        %.5e --- bkw = %.5e --- fwd = %.5e \n",
                    i, err, precbkw[end], bkwall[end], fwdall[end])
            elseif (verbose && do_stats)
                @printf("       ---> it: %2d --- err = %.5e --- prec bkw = \
                        %.5e --- bkw = %.5e \n",
                    i, err, precbkw[end], bkwall[end])
            elseif (verbose)
                @printf("       ---> it: %2d --- err = %.5e \n",
                    i, err)
            end

            if (err <= τ) # Stopping criterion
                yₛ = Hₛ[1:i, 1:i] \ sₛ[1:i]
                if kind == "left"
                    addvec = Vₛ[:, 1:i] * yₛ
                elseif kind == "right"
                    addvec = precond.Mx(Vₛ[:, 1:i] * yₛ, do_stats)[1]
                elseif kind == "flexible"
                    addvec = Zₛ[:, 1:i] * yₛ
                else
                    error("GMRES kind unknown.")
                end
                x = x + xconvert(u, rnrmInf) * xconvert(u, addvec)
                @goto out
            end
        end
        yₛ = Hₛ[1:m, 1:m] \ sₛ[1:m]
        if kind == "left"
            addvec = Vₛ[:, 1:m] * yₛ
        elseif kind == "right"
            addvec = precond.Mx(Vₛ[:, 1:m] * yₛ, do_stats)[1]
        elseif kind == "flexible"
            addvec = Zₛ[:, 1:m] * yₛ
        else
            error("GMRES kind unknown.")
        end
        x = x + xconvert(u, rnrmInf) * xconvert(u, addvec)
        @label out
        push!(stats["gmresits"], nit)
    end

    if (verbose)
        @printf("It GMRES total: %2d \n", nit)
    end

    stats["bkw"] = bkw
    stats["fwd"] = fwd
    pushfirst!(stats["err"], 1) # The first err is missing, add 1 arbitrarily.
    if do_stats
        pushfirst!(precbkw, 1.0) # The first precbkw is missing, add 1 arbitrarily.
        stats["precbkw"] = precbkw
        pushfirst!(bkwall, bkw[1]) # The first bkw is missing, add it.
        stats["bkwall"] = bkwall
        if xexact !== nothing
            pushfirst!(fwdall, fwd[1]) # The first fwd is missing, add it.
        end
        stats["fwdall"] = fwdall
    end

    x, stats
end

# NOTE: Added for the julia LSP to cover the .jl files in the scripts/ folder.
#       It can be safely removed. See: 
#       https://discourse.julialang.org/t/lsp-missing-reference-woes/98231/8
macro ignore(args...) end

@ignore include("../scripts/sparse.jl")
@ignore include("../scripts/dense.jl")

end
