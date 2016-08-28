### GENERALIZED COLUMN OPS

# We want to support arbitrary dimensional arrays.  When data comes in
# N dimensions, we assume it is an array of N-1 dimensional instances
# and the last dimension gives us the instance count.  We will refer
# to the first N-1 dimensions as generalized "columns" of the
# data. These columns are indexed by the last index of an array,
# i.e. column i corresponds to b[:,:,...,i].

# Here are some convenience functions for generalized columns:
# We consider a 1-D array a single column:

csize(a)=(ndims(a)==1 ? size(a) : size(a)[1:end-1])
csize(a,n)=tuple(csize(a)..., n) # size if you had n columns
clength(a)=(ndims(a)==1 ? length(a) : stride(a,ndims(a)))
ccount(a)=(ndims(a)==1 ? 1 : size(a,ndims(a)))
csub(a,i)=(ndims(a)==1 ? error() : sub(a, ntuple(i->(:), ndims(a)-1)..., i))
cget(a,i)=(ndims(a)==1 ? error() : getindex(a, ntuple(i->(:), ndims(a)-1)..., i))
size2(y)=(nd=ndims(y); (nd==1 ? (length(y),1) : (stride(y, nd), size(y, nd)))) # size as a matrix
size2(y,i)=size2(y)[i]

function minibatch(x, y, batchsize)
    data = Any[]
    for i=1:batchsize:ccount(x)
        j=min(i+batchsize-1,ccount(x))
        push!(data, (cget(x,i:j), cget(y,i:j)))
    end
    return data
end

# CSLICE!  Returns a slice of array b, with columns specified in range
# r, using the storage in KUarray a.  The element types need to match,
# but the size of a does not need to match, it is adjusted as
# necessary.  This is used in train and predict to get data from a raw
# array into a KUarray for minibatching.

# cslice!{A,B,T}(a::KUdense{A,T}, b::KUdense{B,T}, r::UnitRange)=cslice!(a,b.arr,r)

# function cslice!{A,T}(a::KUdense{A,T}, b::BaseArray{T}, r::UnitRange)
#     n  = clength(b) * length(r)
#     length(a.ptr) >= n || resize!(a.ptr, int(resizefactor(KUdense)*n+1))
#     b1 = 1 + clength(b) * (first(r) - 1)
#     copysync!(a.ptr, 1, b, b1, n)
#     a.arr = arr(a.ptr, csize(b, length(r)))
#     return a
# end

# # For non-contiguous columns:
# function cslice!{A,T}(a::KUdense{A,T}, b::BaseArray{T}, cols)
#     ncols = length(cols)
#     clen = clength(b)
#     n = clen * ncols
#     length(a.ptr) >= n || resize!(a.ptr, int(resizefactor(KUdense)*n+1))
#     alen = 0
#     for i=1:ncols
#         bidx = (cols[i]-1)*clen + 1
#         copysync!(a.ptr, alen+1, b, bidx, clen)
#         alen += clen
#     end
#     a.arr = arr(a.ptr, csize(b, ncols))
#     return a
# end


function cslice!{T}(a::BaseArray{T}, b::BaseArray{T}, cols)
    ncols = length(cols)
    clen = clength(b)
    n = clen * ncols
    size(a) == csize(b, ncols) || error("Size mismatch")
    alen = 0
    for i=1:ncols
        bidx = (cols[i]-1)*clen + 1
        copysync!(a, alen+1, b, bidx, clen) # t:134
        alen += clen
    end
    return a
end

function cslice!{T}(a::SparseMatrixCSC{T}, b::SparseMatrixCSC{T}, cols)
    bptr = b.colptr
    nz = 0; for i in cols; nz += bptr[i+1]-bptr[i]; end
    a.m = b.m
    a.n = length(cols)
    resize!(a.nzval, nz)
    resize!(a.rowval, nz)
    resize!(a.colptr, a.n+1)
    a.colptr[1] = a1 = aj = 1
    for bj in cols                 # copy column b[:,bj] to a[:,aj]
        b1 = bptr[bj]
        nz = bptr[bj+1]-b1
        copysync!(a.nzval, a1, b.nzval, b1, nz) # t:217
        copysync!(a.rowval, a1, b.rowval, b1, nz) # t:191
        a1 += nz
        a.colptr[aj+=1] = a1
    end
    @assert aj == a.n+1
    return a
end

# TODO: write the non-contiguous sparse version
# function cslice!{A,B,T}(a::KUsparse{A,T}, b::KUsparse{B,T}, r::UnitRange)
#     bptr = cpucopy(b.colptr)
#     nz = 0; for i in r; nz += bptr[i+1]-bptr[i]; end
#     a.m = b.m
#     a.n = length(r)
#     resize!(a.nzval, nz)
#     resize!(a.rowval, nz)
#     aptr = Array(Int32, a.n+1)
#     aptr[1] = a1 = aj = 1
#     for bj in r                 # copy column b[:,bj] to a[:,aj]
#         b1 = bptr[bj]
#         nz = bptr[bj+1]-b1
#         copysync!(a.nzval, a1, b.nzval, b1, nz)
#         copysync!(a.rowval, a1, b.rowval, b1, nz)
#         a1 += nz
#         aptr[aj+=1] = a1
#     end
#     @assert aj == a.n+1
#     a.colptr = convert(A, aptr)
#     return a
# end

# cslice!{A,T}(a::KUsparse{A,T}, b::SparseMatrixCSC{T}, r::UnitRange)=cslice!(a, convert(KUsparse, b), r)

# CCOPY! Copy n columns from src starting at column si, into dst
# starting at column di.  Used by predict to construct output.  
# Don't need the sparse version, output always dense.

# ccopy!{A,T,N}(dst::BaseArray{T,N}, di, src::KUdense{A,T,N}, si=1, n=ccount(src)-si+1)=(ccopy!(dst,di,src.arr,si,n); dst)
# ccopy!{A,B,T,N}(dst::KUdense{A,T,N}, di, src::KUdense{B,T,N}, si=1, n=ccount(src)-si+1)=(ccopy!(dst.arr,di,src.arr,si,n); dst)

function ccopy!{T,N}(dst::BaseArray{T,N}, di, src::BaseArray{T,N}, si=1, n=ccount(src)-si+1)
    @assert csize(dst)==csize(src)
    clen = clength(src)
    d1 = 1 + clen * (di - 1)
    s1 = 1 + clen * (si - 1)
    copysync!(dst, d1, src, s1, clen * n)
    return dst
end

# CADD! Add n columns from src starting at column si, into dst
# starting at column di.  Used by uniq!  Don't need sparse version,
# weights always dense.

using Base.LinAlg: axpy!

# cadd!{A,T,N}(dst::BaseArray{T,N}, di, src::KUdense{A,T,N}, si=1, n=ccount(src)-si+1)=(cadd!(dst,di,src.arr,si,n); dst)
# cadd!{A,B,T,N}(dst::KUdense{A,T,N}, di, src::KUdense{B,T,N}, si=1, n=ccount(src)-si+1)=(cadd!(dst.arr,di,src.arr,si,n); dst)

function cadd!{T,N}(dst::BaseArray{T,N}, di, src::BaseArray{T,N}, si=1, n=ccount(src)-si+1)
    @assert csize(dst)==csize(src)
    @assert ccount(dst) >= di+n-1
    @assert ccount(src) >= si+n-1
    clen = clength(src)
    d1 = 1 + clen * (di - 1)
    s1 = 1 + clen * (si - 1)
    n1 = clen * n
    axpy!(n1, one(T), pointer(src, s1), 1, pointer(dst, d1), 1)
    return dst
end

# CCAT! generalizes append! to multi-dimensional arrays.  Adds the
# ability to specify particular columns to append.  Used in
# kperceptron to add support vectors.

# ccat!{A,B,T,N}(a::KUdense{A,T,N}, b::KUdense{B,T,N}, cols=(1:ccount(b)))=ccat!(a,b.arr,cols)

# function ccat!{A,T,N}(a::KUdense{A,T,N}, b::BaseArray{T,N}, cols=(1:ccount(b)))
#     @assert csize(a)==csize(b)
#     alen = length(a)
#     clen = clength(a)
#     ncols = length(cols)
#     n = alen + ncols * clen
#     length(a.ptr) >= n || resize!(a.ptr, round(Int,resizefactor(KUdense)*n+1))
#     for i=1:ncols
#         bidx = (cols[i]-1)*clen + 1
#         copysync!(a.ptr, alen+1, b, bidx, clen)
#         alen += clen
#     end
#     a.arr = arr(a.ptr, csize(a, ccount(a) + ncols))
#     return a
# end

# ccat!{A,T}(a::KUsparse{A,T}, b::SparseMatrixCSC{T}, cols=(1:ccount(b)))=ccat!(a,convert(KUsparse,b),cols)

# function ccat!{A,B,T}(a::KUsparse{A,T}, b::KUsparse{B,T}, cols=(1:ccount(b)))
#     # a: m, n, colptr, rowval, nzval
#     # colptr[i]: starting index (in rowval,nzval) of column i
#     # colptr[n+1]: nz+1
#     @assert size(a,1) == size(b,1)
#     aptr = cpucopy(a.colptr)
#     bptr = cpucopy(b.colptr)
#     na = aptr[a.n+1]-1          # count new nonzero entries in a
#     ncols = length(cols)
#     for i in cols; na += bptr[i+1]-bptr[i]; end
#     resize!(a.nzval, na)
#     resize!(a.rowval, na)
#     na = aptr[a.n+1]-1          # restart the count
#     for i=1:ncols
#         bj=cols[i]              # bj'th column of b
#         aj=a.n+i                # will become aj'th column of a
#         nz=bptr[bj+1]-bptr[bj]  # with nz nonzero values
#         @assert length(aptr) == aj
#         push!(aptr, aptr[aj]+nz) # aptr[aj+1] = aptr[aj]+nz
#         copysync!(a.nzval,na+1,b.nzval,bptr[bj],nz)
#         copysync!(a.rowval,na+1,b.rowval,bptr[bj],nz)
#         na = na+nz
#     end
#     @assert length(aptr) == a.n + ncols + 1
#     resize!(a.colptr, a.n + ncols + 1)
#     copysync!(a.colptr, a.n+2, aptr, a.n+2, ncols)
#     a.n += ncols
#     return a
# end

### UNIQ! leaves unique columns in its first argument and sums to
### corresponding columns in the remaining arguments.  Used by
### kperceptron in merging identical support vectors.

# function uniq!{A<:Array}(s::KUdense{A}, ww::KUdense...)
#     oldn = ccount(s)                                            # number of original support vectors
#     for w in ww; @assert ccount(w) == oldn; end 
#     ds = Dict{Any,Int}()                                        # support vector => new index
#     newn = 0                                                    # number of new support vectors
#     for oldj=1:oldn
#         newj = get!(ds, _colkey(s,oldj), newn+1)
#         if newj <= newn                                         # s[:,oldj] already in s[:,newj]
#             @assert newj <= newn == length(ds) < oldj
#             for w in ww; cadd!(w,newj,w,oldj,1); end
#         else                                                    # s[:,oldj] to be copied to s[:,newj]                    
#             @assert newj == newn+1 == length(ds) <= oldj	
#             newn += 1
#             if newj != oldj
#                 ccopysync!(s,newj,s,oldj,1)
#                 for w in ww; ccopy!(w,newj,w,oldj,1); end
#             end
#         end
#     end
#     @assert newn == length(ds)
#     resize!(s, csize(s, newn))
#     for w in ww; resize!(w, csize(w, newn)); end
#     return tuple(s, ww...)
# end

# function uniq!{A<:Array}(s::KUsparse{A}, ww::KUdense...)
#     oldn = ccount(s)                                            # number of original support vectors
#     for w in ww; @assert ccount(w) == oldn; end 
#     ds = Dict{Any,Int}()                                        # support vector => new index
#     @assert s.colptr[1]==1
#     ncol = 0
#     nnz = 0
#     for oldj=1:oldn
#         newj = get!(ds, _colkey(s,oldj), ncol+1)
#         if newj <= ncol                                          # s[:,oldj] already in s[:,newj]
#             @assert newj <= length(ds) == ncol < oldj
#             for w in ww; cadd!(w,newj,w,oldj,1); end
#         else                                                    # s[:,oldj] to be copied to s[:,newj]                    
#             @assert newj == ncol+1 == length(ds) <= oldj	
#             from = s.colptr[oldj]
#             nval = s.colptr[oldj+1] - from
#             to = nnz+1
#             ncol += 1
#             nnz += nval
#             if newj != oldj
#                 copysync!(s.rowval, to, s.rowval, from, nval)
#                 copysync!(s.nzval, to, s.nzval, from, nval)
#                 s.colptr[ncol+1] = nnz+1
#                 for w in ww; ccopy!(w,newj,w,oldj,1); end
#             else 
#                 @assert to == from
#                 @assert s.colptr[ncol+1] == nnz+1
#             end
#         end
#     end
#     @assert length(ds) == ncol
#     s.n = ncol
#     resize!(s.colptr, ncol+1)
#     resize!(s.rowval, nnz)
#     resize!(s.nzval,  nnz)
#     for w in ww; resize!(w, csize(w, s.n)); end
#     return tuple(s, ww...)
# end

# _colkey{A<:Array}(s::KUdense{A},j)=sub(s.arr, ntuple(i->(i==ndims(s) ? (j:j) : Colon()), ndims(s))...)

# function _colkey{A<:Array}(s::KUsparse{A},j)
#     a=s.colptr[j]
#     b=s.colptr[j+1]-1
#     r=sub(s.rowval, a:b)
#     v=sub(s.nzval, a:b)
#     (r,v)
# end

# Getting columns one at a time is expensive, just copy the whole array
# CudaArray does not support sub, even if it did we would not be able to hash it
# getcol{T}(s::KUdense{CudaArray,T}, j)=(n=clength(s);copysync!(Array(T,csize(s,1)), 1, s.arr, (j-1)*n+1, n))

# we need to look at the columns, might as well copy

# function uniq!{A<:CudaArray}(s::KUdense{A}, ww::KUdense...)
#     ss = cpucopy(s)
#     uniq!(ss, ww...)
#     cslice!(s, ss, 1:ccount(ss))
#     return tuple(s, ww...)
# end

# function uniq!{A<:CudaArray}(s::KUsparse{A}, ww::KUdense...)
#     ss = cpucopy(s)
#     uniq!(ss, ww...)
#     cslice!(s, ss, 1:ccount(ss))
#     return tuple(s, ww...)
# end



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
#     copysync!(uu, 1, u, 1, size(u,1)*n)
#     copysync!(vv, 1, v, 1, size(v,1)*n)
#     (ss.m, ss.n, ss.colptr, ss.rowval, ss.nzval) = (s.m, s.n, gpucopy(s.colptr), gpucopy(s.rowval), gpucopy(s.nzval))
#     return (ss,uu,vv)
# end

# function cslice!{A,T,I}(a::KUsparse{A,T,I}, b::SparseMatrixCSC{T,I}, r::UnitRange)
#     nz = 0; for i in r; nz += b.colptr[i+1]-b.colptr[i]; end
#     a.m = b.m
#     a.n = length(r)
#     resize!(a.nzval, nz)
#     resize!(a.rowval, nz)
#     aptr = Array(I, a.n+1)
#     aptr[1] = a1 = aj = 1
#     for bj in r                 # copy column b[:,bj] to a[:,aj]
#         b1 = b.colptr[bj]
#         nz = b.colptr[bj+1]-b1
#         copysync!(a.nzval.arr, a1, b.nzval, b1, nz)
#         copysync!(a.rowval.arr, a1, b.rowval, b1, nz)
#         a1 += nz
#         aptr[aj+=1] = a1
#     end
#     @assert aj == a.n+1
#     copysync!(a.colptr, aptr)
#     return a
# end

