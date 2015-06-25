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
