# File: spai.jl
# Author: Bastien Vieublé
# Email: bastien.vieuble@amss.ac.cn
# Github: https://github.com/bvieuble

"""
```julia
M::SparseMatrixCSC{Tv,Ti} = spai(
    A ::Union{SparseMatrixCSC{Tv,Ti},Adjoint{Tv,SparseMatrixCSC{Tv,Ti}}},  # \
        Sparse matrix A we compute the SPAI of 
    ϵ ::Float64,  # Accuracy of the approximation
    α ::Integer,  # Maximum number of time we refine a column 
    β ::Integer   # Maximum number of nnz to add in a column \
                                   after one refinement
) where {Tv<:AbstractFloat,Ti<:Integer} 
```
Computation of Sparse Approximate Inverse preconditioner (SPAI) of a given 
sparse matrix A. This implementation is a Julia rewrite of the Matlab code of this 
[github repo](https://github.com/Noaman67khan/SPAI-GMRES-IR), which is itself
the companion code of the article *"Mixed Precision Iterative Refinement with Sparse
Approximate Inverse Preconditioning"*[^2].

[^2]: Erin Carson & Noaman Khan, *"Mixed Precision Iterative Refinement with 
      Sparse Approximate Inverse Preconditioning"*, SIAM Journal on Scientific 
      Computing, **(2023)**.

In more detail, the function implement the following algorithm from[^2]. Note 
however that, in our implementation, the initial sparsity pattern 
\$\\mathcal{J}\$ is set to be always the identity.

![spai_img](./assets/spai.png)
"""
function spai(
        A::Union{SparseMatrixCSC{Tv,Ti},Adjoint{Tv,SparseMatrixCSC{Tv,Ti}}}, 
        ϵ::Float64,
        α::Integer, 
        β::Integer
    ) where {Tv<:AbstractFloat,Ti<:Integer}

    if (typeof(A) != SparseMatrixCSC{eltype(A),Int64} &&
        typeof(A) != Adjoint{eltype(A),SparseMatrixCSC{eltype(A),Int64}})
        error("A should be a SparseMatrixCSC object.")
    end

    n = size(A)[1];
    𝐽 = sparse(I, n, n);
    M = spzeros(eltype(A), n, n);

    # Loop on the columns
    for k = 1:n
        eₖ = Matrix(I, n, n)[:, k];
        𝐽ₖ = findall(𝐽[:, k]);

        # Refine α times the column
        for _ = 1:α

            # Compute the shadow of Jk
            𝐼ₖ = Int64[];
            for i in sort(unique(A[:, 𝐽ₖ].rowval))
                if sum(abs.(A[i, 𝐽ₖ])) != 0
                    𝐼ₖ = push!(𝐼ₖ, i);
                end
            end

            # Compute the kth column of M
            Aₖ⁻ = A[𝐼ₖ, 𝐽ₖ];
            eₖ⁻ = eₖ[𝐼ₖ];
            # Julia QR facto on sparse matrix leads to instabilities. For this
            # reason, the factorization is performed on a densified matrix.
            # QRtk = qr(Atk); 
            QRₖ = qr(Matrix(Aₖ⁻));

            mₖ⁻ = QRₖ \ eₖ⁻;
            M[𝐽ₖ, k] = mₖ⁻;
            sₖ⁻ = Aₖ⁻ * mₖ⁻ - eₖ⁻;

            # If the residual is satisfying we stop the iterations
            if (norm(sₖ⁻) < ϵ)
                break;
            end

            # If the residual is not satisfying we add nonzeros in mk
            𝐿ₖ = union(𝐼ₖ, k);
 
            𝐽ₖ⁺ = Int64[];
            for ll = 1:size(𝐿ₖ)[1]
                l = 𝐿ₖ[ll];
                𝑁 = Int64[];
                for j in A[l, :].nzind
                    if (A[l, j] != 0)
                        𝑁 = union(𝑁, j);
                    end
                end
                𝐽ₖ⁺ = union(𝐽ₖ⁺, 𝑁);
            end
            𝐽ₖ⁺ = setdiff(𝐽ₖ⁺, 𝐽ₖ);

            # Adding indices
            ρₖ = 0;
            Ρₖ = Float64[];
            Ρₖ_idx = Int64[];
            n1 = norm(sₖ⁻, 2);

            for jj = 1:size(𝐽ₖ⁺)[1]
                j = 𝐽ₖ⁺[jj];

                n2 = norm(Vector(A[𝐼ₖ, j]), 2);
                # n2 = norm(A[Ik,j]);

                ρⱼₖ = sqrt(abs((n1^2 - ((sₖ⁻' * A[𝐼ₖ, j])^2 / (n2^2)))));
                ρₖ = ρₖ + ρⱼₖ;
                push!(Ρₖ, ρⱼₖ);
                push!(Ρₖ_idx, j);
            end
            # Rojk = reduce(vcat, transpose.(Rojk));

            ρₖ = ρₖ / (size(𝐽ₖ⁺)[1]);

            # Select new column nonzeroes to add
            perm = partialsortperm(Ρₖ, 1:min(size(Ρₖ)[1], β));
            for idx = 1:min(size(Ρₖ)[1], β)
                if (Ρₖ[perm][idx] <= ρₖ)
                    j = Ρₖ_idx[perm][idx];
                    𝐽ₖ = union(𝐽ₖ, j);
                    𝐽ₖ⁺ = setdiff(𝐽ₖ⁺, j);
                else
                    break
                end
            end
        end
    end

    return M;
end
