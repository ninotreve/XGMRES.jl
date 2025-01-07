# File: sparse.jl
# Author: Bastien Vieublé
# Email: bastien.vieuble@amss.ac.cn
# Github: https://github.com/bvieuble/XGMRES.jl
# COMMAND: sh scripts/scripts.sh sparse.jl 
# OR julia --project=$PATH_TO_ROOT sparse.jl

using XGMRES, SparseArrays, LinearAlgebra, Quadmath, Random, Printf, BFloat16s,
      MatrixMarket, SuiteSparseMatrixCollection;

# Set the different floating-point precisions 
b = BFloat16;
h = Float16;
s = Float32;
d = Float64;
q = Float128;

# Fetch the sparse matrix
ssmc = ssmc_db(verbose=false)
matrix = ssmc_matrices(ssmc, "", "saylr1")
path = fetch_ssmc(matrix, format="MM")
n = matrix.nrows[1]
A = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))

# Generate the linear system
Aₜ = xconvert(q, A);
_, n = size(A);
Random.seed!(1);
xexactₜ = rand(q, n);
bₜ = Aₜ * xexactₜ;
bₑ = xconvert(d, bₜ);

# Combination of precisions and preconditioning techniques
u, uₓ, uᵣ, uₛ, uₘ, uₐ = [d s q d s d];
kind = "flexible";
bstop = Float64(eps(u)); fstop = Float64(eps(u)) * 100

# Generate the preconditioner
# precond = create_precond_I(A,uₐ);
# precond = create_precond_ilu(A,1e-7,uₓ,uₐ,uₘ);
# precond = create_precond_lu(A,uₓ,uₐ,uₘ);
precond = create_precond_spai(A, uₓ, uₐ, uₘ; kind=kind, ϵ=0.4, β = 8, α=round(Int64, n / 8, RoundUp));
# precond = create_precond_poly(A,bₑ,uₓ,uₐ,uₘ;deg=50);

# Call xgmres 
xcomp, stats = xgmres(Aₜ, bₜ, precond=precond, kind=kind, xexact=xexactₜ,
                      maxrestrt=20, m=n, bstop=bstop, fstop=fstop, 
                      τ=1e-6, verbose=true, u=u, uᵣ=uᵣ, uₛ=uₛ, stop=0.9,
                      do_stats=true, do_κ=false);

# Print errors and condition numbers
xcompₜ = xconvert(q, xcomp);
@printf("final bkw = %5e --- final fwd = %5e \n", norm(Aₜ * xcompₜ - bₜ, Inf) /
        (norm(Aₜ, Inf) * norm(xexactₜ, Inf) + norm(bₜ, Inf)),
        norm(xcompₜ - xexactₜ, Inf) / norm(xexactₜ, Inf));
