using Pkg; Pkg.activate(".")
using XGMRES, LinearAlgebra, MUMPS, MatrixMarket, MPI, SparseArrays

path = expanduser("~/SC25/data/matrix12.mtx")
A = MatrixMarket.mmread(path)

function read_rhs_vector(path::AbstractString)
    lines = readlines(path)
    n = parse(Int, lines[1])
    b = Float64[parse(Float64, lines[i]) for i in 2:n+1]
    return b
end

b_path = expanduser("~/SC25/data/ones12.rhs")
b = read_rhs_vector(b_path)

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