import Base: At_mul_B!, A_mul_B!, similar

itype{Tv,Ti}(::SparseMatrixCSC{Tv,Ti})=Ti
similar{Tv,Ti}(::SparseMatrixCSC{Tv,Ti},m,n)=spzeros(Tv,Ti,m,n) # this is missing

At_mul_B!(k::Matrix, x::SparseMatrixCSC, s::SparseMatrixCSC)=A_mul_B!(k,x',s)

function A_mul_B!(k::Matrix, x::SparseMatrixCSC, s::SparseMatrixCSC) # 1607
    @assert size(k)==(size(x,1), size(s,2))
    fill!(k, zero(eltype(k)))
    @inbounds @simd for scol=1:size(s,2)
        @inbounds @simd for sp=s.colptr[scol]:(s.colptr[scol+1]-1)
            srow = s.rowval[sp]
            sval = s.nzval[sp]  # 133
            @inbounds @simd for xp=x.colptr[srow]:(x.colptr[srow+1]-1)
                xrow = x.rowval[xp] # 63
                xval = x.nzval[xp]  # 217
                yinc = xval * sval  # 245
                k[xrow,scol] += yinc # 789
            end
        end
    end
    return k
end

function A_mul_B!(k::Matrix, x::Matrix, s::SparseMatrixCSC) # 1607
    @assert size(k)==(size(x,1), size(s,2))
    fill!(k, zero(eltype(k)))
    @inbounds @simd for scol=1:size(s,2)
        @inbounds @simd for sp=s.colptr[scol]:(s.colptr[scol+1]-1)
            sval = s.nzval[sp]  # 133
            srow = s.rowval[sp] # xcol
            @inbounds @simd for xrow=1:size(x,1)
                xval = x[xrow,srow]
                yinc = xval * sval  # 245
                k[xrow,scol] += yinc # 789
            end
        end
    end
    return k
end

function hcat!{Tv,Ti<:Integer}(a::SparseMatrixCSC{Tv}, b::SparseMatrixCSC{Tv}, vj::Vector{Ti}, nj::Integer)
    # a: m, n, colptr, rowval, nzval
    # colptr[i]: starting index (in rowval,nzval) of column i
    # colptr[n+1]: nz+1
    @assert size(a,1) == size(b,1)
    @inbounds for i=1:nj
        j = vj[i]  # concat b[:,j]
        b0 = b.colptr[j]
        b1 = b.colptr[j+1]-1
        nz = b1-b0+1
        a.colptr = push!(a.colptr, a.colptr[end]+nz)
        if nz > 0
            a.rowval = append!(a.rowval, b.rowval[b0:b1])
            a.nzval = append!(a.nzval, b.nzval[b0:b1])
        end
    end
    a.n += nj
    return a
end

function uniq!(s::SparseMatrixCSC, u::AbstractArray, v::AbstractArray)
    ds = Dict{Any,Int}()        # dictionary of support vectors
    ns = 0                      # number of support vectors
    s0 = spzeros(eltype(s), Int32, size(s,1), ns) # new sv matrix
    for j=1:size(s,2)
        jj = get!(ds, s[:,j], ns+1)
        if jj <= ns             # s[:,j] already in s0[:,jj]
            @assert ns == length(ds) < j
            u[:,jj] += u[:,j]
            v[:,jj] += v[:,j]
        else                    # s[:,j] to be added to s0
            @assert jj == ns+1 == length(ds) <= j
            ns = ns+1
            hcat!(s0, s, [j], 1)
            if jj != j
                u[:,jj] = u[:,j]
                v[:,jj] = v[:,j]
            end
        end
    end
    @assert ns == length(ds) == size(s0,2)
    u = size!(u, (size(u,1),ns))
    v = size!(v, (size(v,1),ns))
    for f in names(s); s.(f) = s0.(f); end
    return (s,u,v)
end
