### GENERALIZED COLUMN OPS

# We want to support arbitrary dimensional arrays.  When data comes in
# N dimensions, we assume it is an array of N-1 dimensional instances
# and the last dimension gives us the instance count.  We will refer
# to the first N-1 dimensions as generalized "columns" of the
# data. These columns are indexed by the last index of an array,
# i.e. column i corresponds to b[:,:,...,i].


# CSLICE!  Returns a slice of array b, with columns specified in range
# r, using the storage in KUarray a.  The element types need to match,
# but the size of a does not need to match, it is adjusted as
# necessary.  This is used, for example, for getting data from a raw
# array into a KUarray for minibatching.

function cslice!{A,T}(a::KUdense{A,T}, b::Union(Array{T},CudaArray{T}), r::UnitRange)
    n  = clength(b) * length(r)
    length(a.ptr) >= n || resize!(a.ptr, int(resizefactor(KUdense)*n+1))
    b1 = 1 + clength(b) * (first(r) - 1)
    copy!(a.ptr, 1, b, b1, n)
    a.arr = arr(a.ptr, csize(b, length(r)))
    return a
end

function cslice!{A,T}(a::KUsparse{A,T}, b::SparseMatrixCSC{T,Int32}, r::UnitRange)
    nz = 0; for i in r; nz += b.colptr[i+1]-b.colptr[i]; end
    a.m = b.m
    a.n = length(r)
    resize!(a.nzval, nz)
    resize!(a.rowval, nz)
    resize!(a.colptr, 1+a.n)
    a.colptr[1] = a1 = aj = 1
    for bj in r                 # copy column b[:,bj] to a[:,aj]
        b1 = b.colptr[bj]
        nz = b.colptr[bj+1]-b1
        copy!(a.nzval, a1, b.nzval, b1, nz)
        copy!(a.rowval, a1, b.rowval, b1, nz)
        a1 += nz
        a.colptr[aj+=1] = a1
    end
    @assert aj == a.n+1
    return a
end


# CCOPY! Copy n columns from src starting at column si, into dst
# starting at column di.

function ccopy!(dst, di, src::KUdense, si=1, n=ccount(src))
    @assert eltype(dst)==eltype(src)
    @assert csize(dst)==csize(src)
    clen = clength(src)
    d1 = 1 + clen * (di - 1)
    s1 = 1 + clen * (si - 1)
    copy!(dst, d1, src.ptr, s1, clen * n)
    gpusync()
    return dst
end

# CCAT! generalizes append! to multi-dimensional arrays.  Adds the
# ability to specify particular columns to append.

function ccat!(a::KUdense, b, cols=(1:ccount(b)), ncols=length(cols))
    @assert eltype(a)==eltype(b)
    @assert csize(a)==csize(b)
    alen = length(a)
    clen = clength(a)
    n = alen + ncols * clen
    length(a.ptr) >= n || resize!(a.ptr, int(resizefactor(KUdense)*n+1))
    for i=1:ncols
        bidx = (cols[i]-1)*clen + 1
        copy!(a.ptr, alen+1, b, bidx, clen)
        alen += clen
    end
    a.arr = arr(a.ptr, csize(a, ccount(a) + ncols))
    gpusync()
    return a
end

function ccat!(a::KUsparse, b::KUsparse, cols=(1:ccount(b)), ncols=length(cols))
    # a: m, n, colptr, rowval, nzval
    # colptr[i]: starting index (in rowval,nzval) of column i
    # colptr[n+1]: nz+1
    @assert size(a,1) == size(b,1)
    aptr = to_host(a.colptr)
    bptr = to_host(b.colptr)
    na = aptr[a.n+1]-1          # count new nonzero entries in a
    for i in cols; na += bptr[i+1]-bptr[i]; end
    resize!(a.nzval, na)
    resize!(a.rowval, na)
    resize!(a.colptr, a.n + ncols + 1)
    na = aptr[a.n+1]-1          # restart the count
    for i=1:ncols
        bj=cols[i]              # bj'th column of b
        aj=a.n+i                # will become aj'th column of a
        nz=bptr[bj+1]-bptr[bj]  # with nz nonzero values
        @assert length(aptr) == aj
        push!(aptr, aptr[aj]+nz) # aptr[aj+1] = aptr[aj]+nz
        copy!(a.nzval,na+1,b.nzval,bptr[bj],nz)
        copy!(a.rowval,na+1,b.rowval,bptr[bj],nz)
        na = na+nz
    end
    @assert length(aptr) == a.n + ncols + 1
    copy!(a.colptr, a.n+2, aptr, a.n+2, ncols)
    a.n += ncols
    return a
end

