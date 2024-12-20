"""
```julia
a::u = xconvert(
    u::DataType,  # Target floating point arith for the conversion
    num::T        # Real number to convert
) where {T<:AbstractFloat}
```
"""
function xconvert(
        u::DataType,
        num::T
    ) where {T<:AbstractFloat}

    return convert(u, num)
end

"""
```julia
a::Vector{u} = xconvert(
    u::DataType,    # Target floating point arith for the conversion
    vec::Vector{T}  # Vector of real to convert
) where {T<:AbstractFloat}
```
"""
function xconvert(
        u::DataType,
        vec::Vector{T}
    ) where {T<:AbstractFloat}

    if (eltype(vec) != u)
        return convert(Vector{u}, vec)
    else
        return vec
    end
end

"""
```julia
a::Matrix{u} = xconvert(
    u::DataType,    # Target floating point arith for the conversion
    mat::Matrix{T}  # Matrix of real to convert
) where {T<:AbstractFloat}
```
"""
function xconvert(
        u::DataType,
        mat::Matrix{T}
    ) where {T<:AbstractFloat}

    if (eltype(mat) != u)
        return convert(Matrix{u}, mat)
    else
        return mat
    end
end

"""
```julia
a::IncompleteLU.ILUFactorization{u,Int64}) = xconvert(
    u::DataType,                                 # Target floating point \
                                                   arith for the conversion
    mat::IncompleteLU.ILUFactorization{T,Int64}  # IncompleteLU factors \
                                                   structure to convert
) where {T<:AbstractFloat}
```
"""
function xconvert(
        u::DataType,
        mat::IncompleteLU.ILUFactorization{T,Int64}
    ) where {T<:AbstractFloat}

    if (eltype(mat) != u)
        return IncompleteLU.ILUFactorization(xconvert(u, mat.L),
            xconvert(u, mat.U))
    else
        return mat
    end
end

"""
```julia
a::LU{u,Matrix{u},Vector{Int64}}) = xconvert(
    u::DataType,                         # Target floating point arith \
                                           for the conversion
    fact::LU{T,Matrix{T},Vector{Int64}}  # LinearAlgebra.LU factors \
                                           structure to convert
) where {T<:AbstractFloat}
```
"""
function xconvert(
        u::DataType,
        fact::LU{T,Matrix{T},Vector{Int64}}
    ) where {T<:AbstractFloat}

    if (eltype(fact) != u)
        return convert(LU{u,Matrix{u},Vector{Int64}}, fact)
    else
        return fact
    end
end

"""
```julia
a::Symmetric{u,SparseMatrixCSC{u,Int64}} = xconvert(
    u::DataType,                                # Target floating point \
                                                  arith for the conversion
    mat::Symmetric{T,SparseMatrixCSC{T,Int64}}  # Symmetric sparse matrix \
                                                  to convert
) where {T<:AbstractFloat}
```
"""
function xconvert(
        u::DataType,
        mat::Symmetric{T,SparseMatrixCSC{T,Int64}}
    ) where {T<:AbstractFloat}

    if (eltype(mat) != u)
        return convert(Symmetric{u,SparseMatrixCSC{u,Int64}}, mat)
    else
        return mat
    end
end

"""
```julia
a::SparseMatrixCSC{u,Ti} = xconvert(
    u::DataType,                 # Target floating point arith for the \
                                   conversion
    mat::SparseMatrixCSC{Tv,Ti}  # Sparse matrix to convert
) where {Tv<:AbstractFloat,Ti<:Integer}
```
"""
function xconvert(
        u::DataType,
        mat::SparseMatrixCSC{Tv,Ti}
    ) where {Tv<:AbstractFloat,Ti<:Integer}

    if (eltype(mat) != u)
        return convert(SparseMatrixCSC{u,Ti}, mat)
    else
        return mat
    end
end

"""
```julia
a::Adjoint{u,SparseMatrixCSC{u,Int64}} = xconvert(
    u::DataType,                              # Target floating point arith \
                                                for the conversion
    mat::Adjoint{T,SparseMatrixCSC{T,Int64}}  # Adjoint sparse matrix to convert
) where {T<:AbstractFloat}
```
"""
function xconvert(
        u::DataType, 
        mat::Adjoint{T,SparseMatrixCSC{T,Int64}}
    ) where {T<:AbstractFloat}

    if (eltype(mat) != u)
        return convert(Adjoint{u,SparseMatrixCSC{u,Int64}}, mat)
    else
        return mat
    end
end

"""
```julia
a::Vector{Vector{u}} = xconvert(
    u::DataType,            # Target floating point arith for the conversion
    vec::Vector{Vector{T}}  # Vector of vector of real to convert
) where {T<:AbstractFloat}
```
"""
function xconvert(
        u::DataType,
        vec::Vector{Vector{T}}
    ) where {T<:AbstractFloat}

    if (eltype(vec) != u)
        vec_cast = Vector{u}[]
        for k = 1:size(vec)[1]
            push!(vec_cast, xconvert(u, vec[k]))
        end
        return vec_cast
    else
        return vec
    end
end

"""
```julia
a::AlgebraicMultigrid.MultiLevel{
        AlgebraicMultigrid.Pinv{u},
        GaussSeidel{SymmetricSweep},
        GaussSeidel{SymmetricSweep},
        SparseMatrixCSC{u,Int64}, 
        Adjoint{u,SparseMatrixCSC{u,Int64}},
        SparseMatrixCSC{u,Int64}, 
        AlgebraicMultigrid.MultiLevelWorkspace{Vector{u},1}
} = xconvert(
    u::DataType,  # Target floating point arith for the conversion
    mat::AlgebraicMultigrid.MultiLevel{         # AlgebraicMultigrid datatype
                AlgebraicMultigrid.Pinv{T},     # to convert
                GaussSeidel{SymmetricSweep},
                GaussSeidel{SymmetricSweep},
                SparseMatrixCSC{T,Int64}, 
                Adjoint{T,SparseMatrixCSC{T,Int64}},
                SparseMatrixCSC{T,Int64}, 
                AlgebraicMultigrid.MultiLevelWorkspace{Vector{T},1}}
) where {T<:AbstractFloat}
```
"""
function xconvert(
        u::DataType,
        mat::AlgebraicMultigrid.MultiLevel{
            AlgebraicMultigrid.Pinv{T},
            GaussSeidel{SymmetricSweep},
            GaussSeidel{SymmetricSweep},
            SparseMatrixCSC{T,Int64},
            Adjoint{T,SparseMatrixCSC{T,Int64}},
            SparseMatrixCSC{T,Int64},
            AlgebraicMultigrid.MultiLevelWorkspace{Vector{T},1}}
    ) where {T<:AbstractFloat}

    if (eltype(mat.final_A) != u)
        # Cast of the levels entry
        levels = AlgebraicMultigrid.Level{SparseMatrixCSC{u,Int64},
            Adjoint{u,SparseMatrixCSC{u,Int64}},SparseMatrixCSC{u,Int64}}[]
        for k = 1:size(mat.levels)[1]
            push!(levels, AlgebraicMultigrid.Level(xconvert(u, mat.levels[k].A),
                xconvert(u, mat.levels[k].P),
                xconvert(u, mat.levels[k].R)))
        end
        # Cast of the final_A entry
        final_A = xconvert(u, mat.final_A)
        # Cast of the coarse_solver entry
        coarse_solver = AlgebraicMultigrid.Pinv(xconvert(u, mat.coarse_solver.pinvA))
        # Cast of the workspace entry
        workspace = AlgebraicMultigrid.MultiLevelWorkspace{Vector{u},1}(
            xconvert(u, mat.workspace.coarse_xs),
            xconvert(u, mat.workspace.coarse_bs),
            xconvert(u, mat.workspace.res_vecs))
        return AlgebraicMultigrid.MultiLevel(levels, final_A, coarse_solver,
            mat.presmoother, mat.postsmoother,
            workspace)
    else
        return mat
    end
end

"""
```julia
a::ILUZero.ILU0Precon{u,Int64,u}) = xconvert(
    u::DataType,                        # Target floating point \
                                          arith for the conversion
    mat::ILUZero.ILU0Precon{T,Int64,T}  # IncompleteLU factors \
                                          structure to convert
) where {T<:AbstractFloat}
```
"""
function xconvert(
        u::DataType,
        mat::ILUZero.ILU0Precon{T,Int64,T}
    ) where {T<:AbstractFloat}

    if (eltype(mat) != u)
        return ILUZero.ILU0Precon(
            mat.m,
            mat.n,
            mat.l_colptr,
            mat.l_rowval,
            xconvert(u,mat.l_nzval),
            mat.u_colptr,
            mat.u_rowval,
            xconvert(u, mat.u_nzval),
            mat.l_map,
            mat.u_map,
            xconvert(u, mat.wrk)
        )
    else
        return mat
    end
end

"""
```julia
    collect_stats(dict_lists, dict)
```
"""
function collect_stats(dict_lists, dict)
    for (key, value) in dict
        if haskey(dict_lists, key)
            push!(dict_lists[key], value)
        else
            dict_lists[key] = []
            push!(dict_lists[key], value)
        end
    end
end

"""
```julia
isnan(x::AbstractVector)
```
"""
function isnan(x::AbstractVector)
    found = false
    for i = 1:size(x, 1)
        if (isnan(x[i]))
            found = true
            break
        end
    end
    found
end

"""
```
BFloat16(num::Float128)
```
Conversion of quadruple precision number to bfloat number. The conversion is 
not supported natively.
"""
function BFloat16s.BFloat16(num::Float128)
    return BFloat16(Float32(num));
end

function Quadmath.Float128(num::BFloat16)
    return Float128(Float32(num));
end

function Base.trunc(i::Type{Integer}, num::BFloat16)
    return trunc(i, Float32(num));
end
