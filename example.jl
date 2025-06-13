using Pkg; Pkg.activate(".")
using XGMRES, LinearAlgebra, MUMPS, MatrixMarket, MPI, SparseArrays

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

MPI.Init()
mumps = Mumps{Float64}(mumps_unsymmetric, default_icntl, default_cntl64)
associate_matrix!(mumps, A)
factorize!(mumps)
associate_rhs!(mumps, b)
solve!(mumps)
x = get_solution(mumps)
finalize(mumps)
MPI.Finalize()

norm(A * x - b) / norm(b)