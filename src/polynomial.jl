# File: polynomial.jl
# Author: Bastien Vieublé
# Email: bastien.vieuble@amss.ac.cn
# Github: https://github.com/bvieuble

"""
```julia
struct Poly{TA<:AbstractFloat,TP<:AbstractFloat}
    A    ::AbstractMatrix{TA}  # The matrix A 
    H    ::Matrix{TP}  # The Arnoldi Hessenberg matrix 
    g    ::Vector{TP}  # Coefficients of the poly in the Arnoldi basis
    deg  ::Integer     # Degree of the polynomial
end
```
A structure that describes the Arnoldi-based polynomial preconditioner. 
Computing such a polynomial from a matrix \$A\$, a right-hand side \$b\$, and 
a given degree can be done with the method [`polynomial`](@ref). The 
application of the polynomial to a vector is available through 
[`Base.:*`](@ref).
"""
struct Poly{TA<:AbstractFloat,TP<:AbstractFloat}
    A    ::AbstractMatrix{TA}
    H    ::Matrix{TP}
    g    ::Vector{TP}
    deg  ::Integer
end

"""
```julia
𝑃::Poly = xconvert(
    u::DataType,       # Target floating point arith for the conversion
    poly::Poly{TA,TP}  # Polynomial to convert
) where {TA<:AbstractFloat,TP<:AbstractFloat}
```
"""
function xconvert(
        u::DataType,
        poly::Poly{TA,TP}
    ) where {TA<:AbstractFloat,TP<:AbstractFloat}

    return Poly(xconvert(u, poly.A), xconvert(u, poly.H), xconvert(u, poly.g),
                poly.deg);
end

"""
```julia
𝑃::Poly = polynomial(
    A    ::AbstractMatrix{TA},
    b    ::AbstractVector{TB},
    deg  ::Integer
) where {TA<:AbstractFloat,TB<:AbstractFloat}
```
Computation of an Arnoldi-based polynomial preconditioner for the matrix \$A\$.
The floating point arithmetic ``u`` at which the computation is performed is the 
arithmetic of the elements of `A`; if `b` is not provided in this arithmetic, 
it is casted in precision uₓ. This implementation is based on the pseudo-code 
proposed in the article *"Polynomial Preconditioned GMRES and GMRES-DR"*[^3].

![poly_img](./assets/polynomial.png)

[^3]: Quan Liu, Ronald B. Morgan, and Walter Wilcox, *"Polynomial 
      Preconditioned GMRES and GMRES-DR"*, SIAM Journal on Scientific 
      Computing, **(2015)**.
"""
function polynomial(
        A::AbstractMatrix{TA},
        b::AbstractVector{TB}, 
        deg::Integer
    ) where {TA<:AbstractFloat,TB<:AbstractFloat}

    u = eltype(A);

    if (eltype(b) != u)
        b = xconvert(u, b);
    end

    n = size(b, 1);
    V = zeros(u, n, deg + 1);
    H = zeros(u, deg + 1, deg);
    # Gw      = Array{Tuple{LinearAlgebra.Givens{uf},uf},1}(undef,n);
    e₁ = zeros(u, n + 1);
    e₁[1] = 1.0;

    r = b ./ norm(b, 2);
    V[:, 1] = r;
    s = e₁;

    for i = 1:deg

        w = A * V[:, i];

        for k = 1:i
            H[k, i] = w' * V[:, k];
            w = w - H[k, i] * V[:, k];
        end
        H[i+1, i] = norm(w, 2);
        V[:, i+1] = w / H[i+1, i];
        # for k = 1:i-1                           
        #   Hw[:,i] = (Gw[k])[1]*Hw[:,i];
        # end
        # Gw[i]     = givens(Hw[:,i],i,i+1);
        # sw[:]     = (Gw[i])[1] * sw;
        # Hw[i,i]   = (Gw[i])[2];
        # Hw[i+1,i] = 0.0;
        # err       = abs(sw[i+1]); 
    end

    QR = qr(H);
    x = QR \ s[1:deg+1];
    # yw = Hw[1:deg,1:deg] \ sw[1:deg];
    # xw = Vf[:,1:deg]*yw;

    return Poly(A, H, x, deg);
end

"""
```julia
y::AbstractVector = (*)(
    𝑃   ::Poly{TP},           # Polynomial
    v   ::AbstractVector{TV}  # Vector
) where {TP<:AbstractFloat,TM<:AbstractFloat,TV<:AbstractFloat}
```
Application of an Arnoldi-based polynomial to a vector \$p(A) \\times v\$.
"""
function (*)(
        𝑃::Poly{TP},
        v::AbstractVector{TV}
    ) where {TP<:AbstractFloat,TV<:AbstractFloat}

    u = eltype(𝑃.H);
    n = size(𝑃.A, 1);
    W = zeros(u, n, 𝑃.deg);

    if (eltype(v) != u)
        v = xconvert(u, v);
    else
        v = v;
    end

    y = 𝑃.g[1] .* v;
    W[:, 1] = v;
    for j = 1:(𝑃.deg-1)
        t = 𝑃.A * W[:, j];
        for i = 1:j
            t = t - 𝑃.H[i, j] .* W[:, i];
        end
        W[:, j+1] = t ./ 𝑃.H[j+1, j];
        y = y + 𝑃.g[j+1] .* W[:, j+1];
    end

    return y;
end
