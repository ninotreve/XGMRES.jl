using MPI
using MUMPS
using LinearAlgebra
using SparseArrays
using Quadmath
using MatrixMarket

function iterative_refinement_mumps(A::SparseMatrixCSC{Float64,Int},
    b::Vector{Float128};
    max_iter::Int = 10,
    tol::Float64 = 1e-8)

    # 初始化 MPI 和 MUMPS
    MPI.Init()
    mumps = Mumps{Float64}(mumps_unsymmetric, default_icntl, default_cntl64)
    # MUMPS.set_icntl!(mumps,4,0)
    associate_matrix!(mumps, A)
    factorize!(mumps)

    # 初始解 x0 (Float128)
    x = Vector{Float128}(undef, length(b))
    x .= 0

    for iter in 1:max_iter
        # r = b - A * x
        A128 = convert(SparseMatrixCSC{Float128, Int}, A)
        r128 = b - A128 * x

        # 检查收敛
        rnorm = norm(r128)/norm(b)
        println("iter $iter: residual norm = $rnorm")
        if rnorm < tol
            break
        end

        # 用双精度求解 A δ = r
        r64 = Vector{Float64}(r128)
        MUMPS.set_job!(mumps,2)
        associate_rhs!(mumps, r64)
        solve!(mumps)
        X = get_solution(mumps)  # Float64
        δ = X[:,1]

        # x ← x + δ
        x .+= Vector{Float128}(δ)
    end

    finalize(mumps)
    MPI.Finalize()

    return x
end

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
b_float128 =Vector{Float128}(b) # 或你有原始 Float128 b

x128 = iterative_refinement_mumps(A, b_float128; max_iter=5, tol=1e-8)