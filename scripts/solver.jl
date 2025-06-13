# File: sparse.jl
# Author: Bastien Vieublé
# Email: bastien.vieuble@amss.ac.cn
# Github: https://github.com/bvieuble/XGMRES.jl
# COMMAND: sh scripts/scripts.sh solver.jl
# OR julia --project=$PATH_TO_ROOT solver.jl

using XGMRES, SparseArrays, LinearAlgebra, Quadmath, Random, Printf, BFloat16s,
      MatrixMarket, SuiteSparseMatrixCollection;

# Set the different floating-point precisions
b = BFloat16;
h = Float16;
s = Float32;
d = Float64;
q = Float128;

# Fetch the sparse matrix
# 设置问题编号，例如 3 表示 solverchallenge25_03
id = 3
suffix = lpad(id, 2, '0')  # 确保是两位数格式

# 拼接路径
base_path = expanduser("~/SC25/data/")
mtx_path = base_path * "solverchallenge25_$suffix.mtx"
rhs_path = base_path * "solverchallenge25_$suffix.rhs"
# mtx_path = base_path * "matrix12.mtx"
# rhs_path = base_path * "ones12.rhs"

# 读取稀疏矩阵
A = MatrixMarket.mmread(mtx_path)

# 读取右端项向量
function read_rhs_vector(path::AbstractString)
    lines = readlines(path)
    n = parse(Int, lines[1])
    b = Float64[parse(Float64, lines[i]) for i in 2:n+1]
    return b
end

b = read_rhs_vector(rhs_path)

# Combination of precisions and preconditioning techniques
# u: update x
# u_x: compute preconditioner
# u_r: compute residual
# u_m: apply preconditioner
# u_a: apply matrix (A * v)
# u_s: rest of gmres
u, uₓ, uᵣ, uₘ, uₐ, uₛ = [q d q d q q];
kind = "left";
bstop = Float64(eps(u)); fstop = Float64(eps(u)) * 100

# Generate the preconditioner
precond = create_precond_mumps(A,uₓ,uₐ);
n = size(A)[1]

# Call xgmres
x, stats = xgmres(A, b, precond=precond, kind=kind, xexact=nothing,
                      maxrestrt=10, m=nothing, bstop=bstop, fstop=fstop,
                      τ=1e-8, verbose=true, u=u, uᵣ=uᵣ, uₛ=uₛ, stop=nothing,
                      do_stats=true, do_κ=false, maxmem=30 * 1e9);

# Print errors
@printf("final residual = %5e \n", norm(A * x - b) / (norm(b)));
