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
# necessary.  This is used in train and predict to get data from a raw
# array into a KUarray for minibatching.

cslice!{A,B,T}(a::KUdense{A,T}, b::KUdense{B,T}, r::UnitRange)=cslice!(a,b.arr,r)

function cslice!{A,T}(a::KUdense{A,T}, b::BaseArray{T}, r::UnitRange)
    n  = clength(b) * length(r)
    length(a.ptr) >= n || resize!(a.ptr, int(resizefactor(KUdense)*n+1))
    b1 = 1 + clength(b) * (first(r) - 1)
    copy!(a.ptr, 1, b, b1, n)
    a.arr = arr(a.ptr, csize(b, length(r)))
    return a
end

# TODO: what to do with Int64 SparseMatrixCSC?

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
# starting at column di.  Used by predict to construct output.

ccopy!{A,T,N}(dst::BaseArray{T,N}, di, src::KUdense{A,T,N}, si=1, n=ccount(src)-si+1)=ccopy!(dst,di,src.arr,si,n)

function ccopy!{T,N}(dst::BaseArray{T,N}, di, src::BaseArray{T,N}, si=1, n=ccount(src)-si+1)
    @assert csize(dst)==csize(src)
    clen = clength(src)
    d1 = 1 + clen * (di - 1)
    s1 = 1 + clen * (si - 1)
    copy!(dst, d1, src, s1, clen * n)
    return dst
end

# CADD! Add n columns from src starting at column si, into dst
# starting at column di.  Used by uniq!

using Base.LinAlg.BLAS: axpy!

cadd!{A,T,N}(dst::KUdense{A,T,N}, di, src::KUdense{A,T,N}, si=1, n=ccount(src)-si+1)=cadd!(dst.arr,di,src.arr,si,n)

function cadd!{T,N}(dst::CudaArray{T,N}, di, src::CudaArray{T,N}, si=1, n=ccount(src)-si+1)
    @assert csize(dst)==csize(src)
    clen = clength(src)
    d1 = 1 + clen * (di - 1)
    s1 = 1 + clen * (si - 1)
    axpy!(clen * n, one(T), pointer(src, si), 1, pointer(dst, di), 1)
    return dst
end

function cadd!{T,N}(dst::Array{T,N}, di, src::Array{T,N}, si=1, n=ccount(src)-si+1)
    @assert csize(dst)==csize(src)
    clen = clength(src)
    d0 = clen * (di - 1)
    s0 = clen * (si - 1)
    n0 = clen * n
    for i=1:n0; dst[d0+i] += src[s0+i]; end
    return dst
end

# CCAT! generalizes append! to multi-dimensional arrays.  Adds the
# ability to specify particular columns to append.  Used in
# kperceptron to add support vectors.

function ccat!{A,B,T,N}(a::KUdense{A,T,N}, b::KUdense{B,T,N}, cols=(1:ccount(b)), ncols=length(cols))
    @assert csize(a)==csize(b)
    alen = length(a)
    clen = clength(a)
    n = alen + ncols * clen
    length(a.ptr) >= n || resize!(a.ptr, int(resizefactor(KUdense)*n+1))
    for i=1:ncols
        bidx = (cols[i]-1)*clen + 1
        copy!(a.ptr, alen+1, b.ptr, bidx, clen)
        alen += clen
    end
    a.arr = arr(a.ptr, csize(a, ccount(a) + ncols))
    return a
end

function ccat!{A,B,T}(a::KUsparse{A,T}, b::KUsparse{B,T}, cols=(1:ccount(b)), ncols=length(cols))
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

### UNIQ! leaves unique columns in its first argument and sums to
### corresponding columns in the remaining arguments.  Used by
### kperceptron in merging identical support vectors.

function uniq!(s::KUdense{Array}, ww::KUdense...)
    oldn = ccount(s)                                            # number of original support vectors
    for w in ww; @assert ccount(ww) == oldn; end 
    ds = Dict{Any,Int}()                                        # support vector => new index
    newn = 0                                                    # number of new support vectors
    for oldj=1:oldn
        newj = get!(ds, getcol(s,oldj), newn+1)
        if newj <= newn                                         # s[:,oldj] already in s[:,newj]
            @assert newj <= newn == length(ds) < oldj
            for w in ww; cadd!(w,newj,w,oldj,1); end
        else                                                    # s[:,oldj] to be copied to s[:,newj]                    
            @assert newj == newn+1 == length(ds) <= oldj	
            newn += 1
            if newj != oldj
                ccopy!(s,newj,s,oldj,1)
                for w in ww; ccopy!(w,newj,w,oldj,1); end
            end
        end
    end
    @assert newn == length(ds)
    resize!(s, csize(s, newn))
    for w in ww; resize!(w, csize(w, newn)); end
    return (s, ww...)
end

function uniq!(s::KUdense{CudaArray}, ww::KUdense...)
    ss = cpucopy(s)                                             # we need to look at the columns, might as well copy
    uniq!(ss, ww...)
    cslice!(s, ss, 1:ccount(ss))
    return (s, ww...)
end

getcol(s::KUdense{Array},j)=sub(s.arr, ntuple(i->(i==ndims(s) ? (j:j) : Colon()), ndims(s))...)

# Getting columns one at a time is expensive, just copy the whole array
# CudaArray does not support sub, even if it did we would not be able to hash it
# getcol{T}(s::KUdense{CudaArray,T}, j)=(n=clength(s);copy!(Array(T,csize(s,1)), 1, s.arr, (j-1)*n+1, n))

# TODO: fix array types

    # ds = Dict{Any,Int}()        # dictionary of support vectors
    # ns = 0                      # number of support vectors
    # s0 = spzeros(eltype(s), Int32, size(s,1), ns) # new sv matrix
    # for j=1:size(s,2)
    #     jj = get!(ds, s[:,j], ns+1)
    #     if jj <= ns             # s[:,j] already in s0[:,jj]
    #         @assert ns == length(ds) < j
    #         u[:,jj] += u[:,j]
    #         v[:,jj] += v[:,j]
    #     else                    # s[:,j] to be added to s0
    #         @assert jj == ns+1 == length(ds) <= j
    #         ns = ns+1
    #         hcat!(s0, s, [j], 1)
    #         if jj != j
    #             u[:,jj] = u[:,j]
    #             v[:,jj] = v[:,j]
    #         end
    #     end
    # end
    # @assert ns == length(ds) == size(s0,2)
    # u = size!(u, (size(u,1),ns))
    # v = size!(v, (size(v,1),ns))
    # for f in names(s); s.(f) = s0.(f); end
    # return (s,u,v)

# function uniq!(s::SparseMatrixCSC, u::AbstractArray, v::AbstractArray)
#     ds = Dict{Any,Int}()        # dictionary of support vectors
#     ns = 0                      # number of support vectors
#     s0 = spzeros(eltype(s), Int32, size(s,1), ns) # new sv matrix
#     for j=1:size(s,2)
#         jj = get!(ds, s[:,j], ns+1)
#         if jj <= ns             # s[:,j] already in s0[:,jj]
#             @assert ns == length(ds) < j
#             u[:,jj] += u[:,j]
#             v[:,jj] += v[:,j]
#         else                    # s[:,j] to be added to s0
#             @assert jj == ns+1 == length(ds) <= j
#             ns = ns+1
#             hcat!(s0, s, [j], 1)
#             if jj != j
#                 u[:,jj] = u[:,j]
#                 v[:,jj] = v[:,j]
#             end
#         end
#     end
#     @assert ns == length(ds) == size(s0,2)
#     u = size!(u, (size(u,1),ns))
#     v = size!(v, (size(v,1),ns))
#     for f in names(s); s.(f) = s0.(f); end
#     return (s,u,v)
# end

# function uniq!(ss::CudaSparseMatrixCSC, uu::AbstractCudaArray, vv::AbstractCudaArray)
#     (s,u,v)=map(cpucopy,(ss,uu,vv))
#     (s,u,v)=uniq!(s,u,v)
#     n = size(s,2)
#     uu = size!(uu, (size(u,1),n))
#     vv = size!(vv, (size(v,1),n))
#     copy!(uu, 1, u, 1, size(u,1)*n)
#     copy!(vv, 1, v, 1, size(v,1)*n)
#     (ss.m, ss.n, ss.colptr, ss.rowval, ss.nzval) = (s.m, s.n, gpucopy(s.colptr), gpucopy(s.rowval), gpucopy(s.nzval))
#     return (ss,uu,vv)
# end

