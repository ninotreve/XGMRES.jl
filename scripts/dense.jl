# File: dense.jl
# Author: Bastien Vieublé
# Email: bastien.vieuble@amss.ac.cn
# Github: https://github.com/bvieuble/XGMRES.jl
# COMMAND: sh scripts/scripts.sh dense.jl 
# OR julia --project=$PATH_TO_ROOT dense.jl

using XGMRES, Quadmath, Random, LinearAlgebra, Printf, BFloat16s;

# Set the different floating-point precisions 
b = BFloat16;
h = Float16;
s = Float32;
d = Float64;
q = Float128;

# Prepare the data
n = 50;
κₐ, κₘ = (1e7, 1e7);
A, M = gen_mat_with_prec(n, κₐ, κₘ);
Aₜ = xconvert(q, A);
Random.seed!(1);
xexactₜ = 1000*rand(q, size(Aₜ, 1));
bₜ = Aₜ * xexactₜ;

# Combination of precisions and preconditioning techniques
u, uᵣ, uₛ, uₘ, uₐ = [d q s s s];
kind = "flexible";
bstop = Float64(eps(u)); fstop = Float64(eps(u)) * 100

# Generate the preconditioner
precond = create_precond_rand(A, M, uₐ, uₘ);

# Call xgmres
xcomp, stats = xgmres(Aₜ, bₜ, precond=precond, kind=kind, xexact=xexactₜ,
                      maxrestrt=20, m=n, bstop=bstop, fstop=fstop, 
                      τ=1e-6, verbose=true, u=u, uᵣ=uᵣ, uₛ=uₛ, stop=0.9, 
                      do_stats=true, do_κ=true);

# Print errors and condition numbers
xcompₜ = xconvert(q, xcomp);
@printf("final bkw = %5e --- final fwd = %5e \n", norm(Aₜ * xcompₜ - bₜ, Inf) /
        (norm(Aₜ, Inf) * norm(xexactₜ, Inf) + norm(bₜ, Inf)), 
        norm(xcompₜ - xexactₜ, Inf) / norm(xexactₜ, Inf));
