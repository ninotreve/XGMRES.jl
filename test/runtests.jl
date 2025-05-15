using XGMRES, Quadmath, Random, LinearAlgebra, Test, BFloat16s,
    AlgebraicMultigrid, SuiteSparseMatrixCollection, MatrixMarket;

const b = BFloat16;
const h = Float16;
const s = Float32;
const d = Float64;
const q = Float128;
const n = 25;

@testset "XGMRES.jl" begin
    ## TEST ON DENSE PROBLEMS #################################################
    # Generate random problems
    rand_pbs = Tuple[]

    κA1, κM1 = (1e2, 1e0)
    A1, M1 = gen_mat_with_prec(n, κA1, κM1)
    A1ₜ = XGMRES.xconvert(q, A1)
    Random.seed!(1)
    xexact1ₜ = rand(q, n)
    b1ₜ = A1ₜ * xexact1ₜ
    push!(rand_pbs, (A1, M1, A1ₜ, xexact1ₜ, b1ₜ, κA1, κM1))

    κA2, κM2 = (1e4, 1e0)
    A2, M2 = gen_mat_with_prec(n, κA2, κM2)
    A2ₜ = XGMRES.xconvert(q, A2)
    Random.seed!(1)
    xexact2ₜ = rand(q, n)
    b2ₜ = A2ₜ * xexact2ₜ
    push!(rand_pbs, (A2, M2, A2ₜ, xexact2ₜ, b2ₜ, κA2, κM2))

    κA3, κM3 = (1e8, 1e0)
    A3, M3 = gen_mat_with_prec(n, κA3, κM3)
    A3ₜ = XGMRES.xconvert(q, A3)
    Random.seed!(1)
    xexact3ₜ = rand(q, n)
    b3ₜ = A3ₜ * xexact3ₜ
    push!(rand_pbs, (A3, M3, A3ₜ, xexact3ₜ, b3ₜ, κA3, κM3))

    κA4, κM4 = (1e8, 1e4)
    A4, M4 = gen_mat_with_prec(n, κA4, κM4)
    A4ₜ = XGMRES.xconvert(q, A4)
    Random.seed!(1)
    xexact4ₜ = rand(q, n)
    b4ₜ = A4ₜ * xexact4ₜ
    push!(rand_pbs, (A4, M4, A4ₜ, xexact4ₜ, b4ₜ, κA4, κM4))

    κA5, κM5 = (1e8, 1e8)
    A5, M5 = gen_mat_with_prec(n, κA5, κM5)
    A5ₜ = XGMRES.xconvert(q, A5)
    Random.seed!(1)
    xexact5ₜ = rand(q, n)
    b5ₜ = A5ₜ * xexact5ₜ
    push!(rand_pbs, (A5, M5, A5ₜ, xexact5ₜ, b5ₜ, κA5, κM5))

    κA6, κM6 = (1e12, 1e0)
    A6, M6 = gen_mat_with_prec(n, κA6, κM6)
    A6ₜ = XGMRES.xconvert(q, A6)
    Random.seed!(1)
    xexact6ₜ = rand(q, n)
    b6ₜ = A6ₜ * xexact6ₜ
    push!(rand_pbs, (A6, M6, A6ₜ, xexact6ₜ, b6ₜ, κA6, κM6))

    κA7, κM7 = (1e12, 1e6)
    A7, M7 = gen_mat_with_prec(n, κA7, κM7)
    A7ₜ = XGMRES.xconvert(q, A7)
    Random.seed!(1)
    xexact7ₜ = rand(q, n)
    b7ₜ = A7ₜ * xexact7ₜ
    push!(rand_pbs, (A7, M7, A7ₜ, xexact7ₜ, b7ₜ, κA7, κM7))

    κA8, κM8 = (1e12, 1e12)
    A8, M8 = gen_mat_with_prec(n, κA8, κM8)
    A8ₜ = XGMRES.xconvert(q, A8)
    Random.seed!(1)
    xexact8ₜ = rand(q, n)
    b8ₜ = A8ₜ * xexact8ₜ
    push!(rand_pbs, (A8, M8, A8ₜ, xexact8ₜ, b8ₜ, κA8, κM8))

    # Test #1
    u, uᵣ, uₛ, uₘ, uₐ = [d q d d d]
    for pb in rand_pbs
        A, _, Aₜ, xexactₜ, bₜ, _, _ = pb

        precond = create_precond_I(A, uₐ)
        xcomp, stats = xgmres(Aₜ, bₜ, precond=precond, kind="left",
            xexact=xexactₜ, maxrestrt=10, m=n, bstop=eps(u),
            fstop=eps(u), τ=1e-16, verbose=false, u=u, uᵣ=uᵣ,
            uₛ=uₛ, stop=nothing, do_stats=false)

        xcompₜ = XGMRES.xconvert(q, xcomp)
        bkw_out = norm(Aₜ * xcompₜ - bₜ, Inf) / (norm(Aₜ, Inf) *
                                                 norm(xexactₜ, Inf) + norm(bₜ, Inf))
        fwd_out = norm(xcompₜ - xexactₜ, Inf) / norm(xexactₜ, Inf)
        @test stats["bkw"][end] < eps(u) && stats["fwd"][end] < eps(u) &&
              bkw_out < eps(u) && fwd_out < eps(u)
    end

    # Test #2
    u, uᵣ, uₛ, uₘ, uₐ = [d q s d d]
    for pb in rand_pbs
        A, M, Aₜ, xexactₜ, bₜ, κₐ, κₘ = pb

        if κₐ <= 1e8
            precond = create_precond_rand(A, M, uₐ, uₘ)
            xcomp, stats = xgmres(Aₜ, bₜ, precond=precond, kind="left",
                xexact=xexactₜ, maxrestrt=50, m=size(bₜ)[1],
                bstop=eps(u), fstop=eps(u), τ=1e-16,
                verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ,
                stop=nothing, do_stats=false)

            xcompₜ = XGMRES.xconvert(q, xcomp)
            bkw_out = norm(Aₜ * xcompₜ - bₜ, Inf) / (norm(Aₜ, Inf) *
                                                     norm(xexactₜ, Inf) + norm(bₜ, Inf))
            fwd_out = norm(xcompₜ - xexactₜ, Inf) / norm(xexactₜ, Inf)
            @test stats["bkw"][end] < eps(u) && stats["fwd"][end] < eps(u) &&
                  bkw_out < eps(u) && fwd_out < eps(u)
        end
    end

    # Test #3
    u, uᵣ, uₛ, uₘ, uₐ = [d q s d d]
    for pb in rand_pbs
        A, M, Aₜ, xexactₜ, bₜ, κₐ, κₘ = pb

        if κₐ <= 1e8
            precond = create_precond_rand(A, M, uₐ, uₘ)
            xcomp, stats = xgmres(Aₜ, bₜ, precond=precond, kind="right",
                xexact=xexactₜ, maxrestrt=50, m=size(bₜ)[1],
                bstop=eps(u), fstop=eps(u), τ=1e-16,
                verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ, stop=nothing,
                do_stats=false)

            xcompₜ = XGMRES.xconvert(q, xcomp)
            bkw_out = norm(Aₜ * xcompₜ - bₜ, Inf) / (norm(Aₜ, Inf) *
                                                     norm(xexactₜ, Inf) + norm(bₜ, Inf))
            fwd_out = norm(xcompₜ - xexactₜ, Inf) / norm(xexactₜ, Inf)
            @test stats["bkw"][end] < eps(u) && stats["fwd"][end] < eps(u) &&
                  bkw_out < eps(u) && fwd_out < eps(u)
        end
    end

    ## TEST ON SPARSE PROBLEMS ################################################
    # Test #1
    ssmc = ssmc_db(verbose=false)
    matrix = ssmc_matrices(ssmc, "", "arc130")
    path = fetch_ssmc(matrix, format="MM")
    A1 = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
    A1ₜ = xconvert(q, A1)
    nA1 = size(A1)[1]
    Random.seed!(1)
    xexact1ₜ = rand(q, nA1)
    b1ₜ = A1ₜ * xexact1ₜ

    u, uₓ, uᵣ, uₛ, uₘ, uₐ = [d d q d d d]

    precond = create_precond_I(A1, uₐ)
    xcomp, stats = xgmres(A1ₜ, b1ₜ, precond=precond, kind="left", xexact=xexact1ₜ,
        maxrestrt=10, m=nA1, bstop=eps(u), fstop=eps(u), τ=1e-8,
        verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ, stop=nothing,
        do_stats=true)
    xcompₜ = XGMRES.xconvert(q, xcomp)
    bkw_out = norm(A1ₜ * xcompₜ - b1ₜ, Inf) / (norm(A1ₜ, Inf) *
                                               norm(xexact1ₜ, Inf) + norm(b1ₜ, Inf))
    fwd_out = norm(xcompₜ - xexact1ₜ, Inf) / norm(xexact1ₜ, Inf)
    @test stats["bkwall"][end] < eps(u) && stats["fwdall"][end] < eps(u) &&
          bkw_out < eps(u) && fwd_out < eps(u)

    # Test #2
    xcomp, stats = xgmres(A1ₜ, b1ₜ, precond=precond, kind="left", xexact=xexact1ₜ,
        maxrestrt=1, m=nA1, bstop=eps(u), fstop=eps(u), τ=1e-8,
        verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ, stop=nothing,
        do_stats=true)
    xcompₜ = XGMRES.xconvert(q, xcomp)
    bkw_out = norm(A1ₜ * xcompₜ - b1ₜ, Inf) / (norm(A1ₜ, Inf) *
                                               norm(xexact1ₜ, Inf) + norm(b1ₜ, Inf))
    fwd_out = norm(xcompₜ - xexact1ₜ, Inf) / norm(xexact1ₜ, Inf)
    @test stats["bkwall"][end] > eps(u) && stats["fwdall"][end] > eps(u) &&
          bkw_out > eps(u) && fwd_out > eps(u)

    ## ILU ##
    # Test #3
    precond = create_precond_ilu(A1, 0.1, uₓ, uₐ, uₘ)
    xcomp, stats = xgmres(A1ₜ, b1ₜ, precond=precond, kind="left", xexact=xexact1ₜ,
        maxrestrt=3, m=nA1, bstop=eps(u), fstop=eps(u), τ=1e-8,
        verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ, stop=nothing,
        do_stats=true)
    xcompₜ = XGMRES.xconvert(q, xcomp)
    bkw_out = norm(A1ₜ * xcompₜ - b1ₜ, Inf) / (norm(A1ₜ, Inf) *
                                               norm(xexact1ₜ, Inf) + norm(b1ₜ, Inf))
    fwd_out = norm(xcompₜ - xexact1ₜ, Inf) / norm(xexact1ₜ, Inf)
    @test stats["bkwall"][end] < eps(u) && stats["fwdall"][end] < eps(u) &&
          bkw_out < eps(u) && fwd_out < eps(u)

    # Test #4
    xcomp, stats = xgmres(A1ₜ, b1ₜ, precond=precond, kind="right", xexact=xexact1ₜ,
        maxrestrt=4, m=nA1, bstop=eps(u), fstop=eps(u), τ=1e-8,
        verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ, stop=nothing,
        do_stats=true)
    xcompₜ = XGMRES.xconvert(q, xcomp)
    bkw_out = norm(A1ₜ * xcompₜ - b1ₜ, Inf) / (norm(A1ₜ, Inf) *
                                               norm(xexact1ₜ, Inf) + norm(b1ₜ, Inf))
    fwd_out = norm(xcompₜ - xexact1ₜ, Inf) / norm(xexact1ₜ, Inf)
    @test stats["bkwall"][end] < eps(u) && stats["fwdall"][end] < eps(u) &&
          bkw_out < eps(u) && fwd_out < eps(u)

    # Test #5
    u, uₓ, uᵣ, uₛ, uₘ, uₐ = [d b q s s s]
    precond = create_precond_ilu(A1, 0.001, uₓ, uₐ, uₘ)
    xcomp, stats = xgmres(A1ₜ, b1ₜ, precond=precond, kind="left", xexact=xexact1ₜ,
        maxrestrt=5, m=nA1, bstop=eps(u), fstop=eps(u), τ=1e-8,
        verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ, stop=nothing,
        do_stats=true)
    xcompₜ = XGMRES.xconvert(q, xcomp)
    bkw_out = norm(A1ₜ * xcompₜ - b1ₜ, Inf) / (norm(A1ₜ, Inf) *
                                               norm(xexact1ₜ, Inf) + norm(b1ₜ, Inf))
    fwd_out = norm(xcompₜ - xexact1ₜ, Inf) / norm(xexact1ₜ, Inf)
    @test stats["bkwall"][end] < eps(u) && stats["fwdall"][end] < eps(u) &&
          bkw_out < eps(u) && fwd_out < eps(u)

    ## LU ## 
    # Test #6
    matrix = ssmc_matrices(ssmc, "", "1138_bus")
    path = fetch_ssmc(matrix, format="MM")
    A2 = MatrixMarket.mmread(joinpath(path[1], "$(matrix.name[1]).mtx"))
    A2ₜ = xconvert(q, A2)
    nA2 = size(A2)[1]
    Random.seed!(1)
    xexact2ₜ = rand(q, nA2)
    b2ₜ = A2ₜ * xexact2ₜ

    u, uₓ, uᵣ, uₛ, uₘ, uₐ = [d s q d d d]

    precond = create_precond_lu(A2, uₓ, uₐ, uₘ)
    xcomp, stats = xgmres(A2ₜ, b2ₜ, precond=precond, kind="right", xexact=xexact2ₜ,
        maxrestrt=4, m=nA2, bstop=eps(u), fstop=eps(u), τ=1e-4,
        verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ, stop=nothing,
        do_stats=true)
    xcompₜ = xconvert(q, xcomp)
    bkw_out = norm(A2ₜ * xcompₜ - b2ₜ, Inf) / (norm(A2ₜ, Inf) *
                                               norm(xexact2ₜ, Inf) + norm(b2ₜ, Inf))
    fwd_out = norm(xcompₜ - xexact2ₜ, Inf) / norm(xexact2ₜ, Inf)
    @test stats["bkwall"][end] < eps(u) && stats["fwdall"][end] < eps(u) &&
          bkw_out < eps(u) && fwd_out < eps(u)

    ## AMG ##
    # Test #7
    A3 = AlgebraicMultigrid.poisson(1000)
    A3ₜ = xconvert(q, A3)
    nA3 = size(A3)[1]
    Random.seed!(1)
    xexact3ₜ = rand(q, nA3)
    b3ₜ = A3ₜ * xexact3ₜ

    u, uₓ, uᵣ, uₛ, uₘ, uₐ = [d s q d d d]

    precond = create_precond_amg(A3, uₓ, uₐ, uₘ)
    xcomp, stats = xgmres(A3ₜ, b3ₜ, precond=precond, kind="left", xexact=xexact3ₜ,
        maxrestrt=4, m=nA3, bstop=eps(u), fstop=eps(u), τ=1e-4,
        verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ, stop=nothing,
        do_stats=true)
    xcompₜ = xconvert(q, xcomp)
    bkw_out = norm(A3ₜ * xcompₜ - b3ₜ, Inf) / (norm(A3ₜ, Inf) *
                                               norm(xexact3ₜ, Inf) + norm(b3ₜ, Inf))
    fwd_out = norm(xcompₜ - xexact3ₜ, Inf) / norm(xexact3ₜ, Inf)
    @test stats["bkwall"][end] < eps(u) && stats["fwdall"][end] < eps(u) &&
          bkw_out < eps(u) && fwd_out < eps(u)

    ## TEST THE LOGS ##########################################################
    # Set the parameters
    u, uᵣ, uₛ, uₘ, uₐ = [d q s d d]
    A, M, Aₜ, xexactₜ, bₜ, κₐ, κₘ = rand_pbs[4]

    # Exact solution not provided
    precond = create_precond_rand(A, M, uₐ, uₘ)
    xcomp, stats = xgmres(Aₜ, bₜ, precond=precond, kind="left",
        xexact=xexactₜ, maxrestrt=50, m=size(bₜ)[1],
        bstop=eps(u), fstop=eps(u), τ=1e-16,
        verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ,
        stop=nothing, do_stats=false)
    @test size(stats["bkw"], 1) == size(stats["fwd"], 1) ==
          size(stats["gmresits"], 1)

    # Exact solution not provided
    xcomp, stats = xgmres(Aₜ, bₜ, precond=precond, kind="left",
        maxrestrt=50, m=size(bₜ)[1],
        bstop=eps(u), fstop=eps(u), τ=1e-16,
        verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ,
        stop=nothing, do_stats=false)
    @test size(stats["fwd"], 1) == 0

    # do_stats parameter activated 
    xcomp, stats = xgmres(Aₜ, bₜ, precond=precond, kind="left",
        xexact=xexactₜ, maxrestrt=50, m=size(bₜ)[1],
        bstop=eps(u), fstop=eps(u), τ=1e-16,
        verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ,
        stop=nothing, do_stats=true, do_κ=true)
    @test size(stats["bkw"])[1] == size(stats["fwd"])[1]
    @test size(stats["bkwall"], 1) == size(stats["fwdall"], 1) ==
          size(stats["err"], 1) == stats["gmresits"][end] + 1

    # do_stats parameter activated and exact solution not provided
    xcomp, stats = xgmres(Aₜ, bₜ, precond=precond, kind="left",
        maxrestrt=50, m=size(bₜ)[1],
        bstop=eps(u), fstop=eps(u), τ=1e-16,
        verbose=false, u=u, uᵣ=uᵣ, uₛ=uₛ,
        stop=nothing, do_stats=true, do_κ=true)
    @test size(stats["fwd"], 1) == size(stats["fwdall"], 1) == 0
    @test size(stats["bkwall"], 1) == size(stats["err"], 1) ==
          stats["gmresits"][end] + 1
end
