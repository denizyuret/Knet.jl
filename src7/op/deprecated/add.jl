"""
@knet Op add(x1,x2; alpha=1,beta=1) performs element-wise broadcasting
addition.  alpha and beta are used to scale x1 and x2 respectively.
The result computed is y = alpha x1 + beta x2.  The size of the output
y always matches the size of x2.  Broadcasting is performed as follows
(following Julia convention, size below refers to the tuple of
dimensions, fastest changing first, and ndims refers to the number of
dimensions):

- If x1 and x2 are the same size their elements are added.

- If x1 and x2 have the same ndims but different sizes, each dimension
  of the x1 array must match the corresponding dimension of the x2
  array or must be equal to 1. In the latter case, the same value from
  the x1 array for those dimensions will be used to blend into the x2
  array.  (At least this is what it says in the NVIDIA cudnnAddTensor
  doc but unfortunately not all combinations work as promised.  I sent
  them a bug report and wrote my own kernels to fill in the gaps for
  now -- Oct 16, 2015.)
  Example: (5,4,1,1)+(5,4,3,2)=>(5,4,3,2)

- If x1 and x2 have different ndims, x1 is assumed to be missing its
  rightmost dims and those are assumed to be 1.  Example:
  (5,4)+(5,4,3,2)=>(5,4,1,1)+(5,4,3,2)=>(5,4,3,2).

- The one exception to the last rule is when x1 has ndims=1 and x2 has
  ndims > 1.  In this case x1 is matched to the next to last dimension
  of x2.  Example: (3,)+(5,4,3,2)=>(1,1,3,1)+(5,4,3,2)=>(5,4,3,2).

"""
type Add <: Op; alpha; beta; Add(;alpha=1,beta=1,o...)=new(alpha,beta); end

ninputs(::Add)=2
canoverwrite(::Add)=true
back_reads_x(::Add)=false
back_reads_y(::Add)=false

function forw(a::Add, x1, x2, y; o...)
    @assert x2 == nothing || size(y) == size(x2)
    if x1!=nothing && x2!=nothing # we use nothing to represent the zero array
        baddforw!(a.alpha,x1,a.beta,x2,y)
    elseif x2 != nothing
        y===x2 || copysync!(y, x2)
        a.beta==1 || scale!(y, a.beta)
    elseif x1 != nothing && size(y)==size(x1)
        y===x1 || copysync!(y, x1)
        a.alpha==1 || scale!(y, a.alpha)
    else
        y=nothing
    end
    gpusync()
    return y
end

function back(a::Add, dy, dx1, dx2; o...)
    if dx2 != nothing
        size(dx2) == size(dy) || throw(DimensionMismatch("The size of the output must match x2"))
        dx2 === dy  || copysync!(dx2, dy)
        a.beta == 1 || scale!(a.beta, dx2)
    end
    if dx1 != nothing
        baddback!(dy, dx1)
        a.alpha == 1 || scale!(a.alpha, dx1)
    end
    gpusync()
end

addforw0!(alpha,a,beta,b,c)=(a===c||b===c||copysync!(c,b))
baddforw0!(alpha,a,beta,b,c)=(b===c||copysync!(c,b))
baddback0!(alpha,dy,db)=fillsync!(db,0)

function baddforw!(alpha,a,beta,b,c)
    size(c) == size(b) || throw(DimensionMismatch("The size of the output must match the second input"))
    size(a) == size(b) || (a = reshape_to_match(a,b; CUDNN_ADD_SAME_C=true))
    if size(a)==size(b)
        addforw3!(alpha,a,beta,b,c)
    elseif cudnnAddTensorCompatible(a,b)
        baddforw1!(alpha,a,beta,b,c)
    else
        baddforw2!(alpha,a,beta,b,c)
    end
end    

@gpu function addforw1!{T}(alpha::Number,a::CudaArray{T},beta::Number,b::CudaArray{T},c::CudaArray{T})
    c===b || copysync!(c,b)
    cudnnAddTensor(a,c; alpha=T(alpha), beta=T(beta))
    gpusync(); return c
end

@gpu function addforw2!{T}(alpha::Number,a::CudaArray{T},beta::Number,b::CudaArray{T},c::CudaArray{T})
    CUBLAS.geam!('N','N',T(alpha),a,T(beta),b,c)
    gpusync(); return c
end

function addforw3!(alpha,a,beta,b,c)
    if c===b
        beta != 1 && scale!(beta, c)
        axpy!(alpha, a, c)
    elseif c===a
        alpha != 1 && scale!(alpha, c)
        axpy!(beta, b, c)
    else
        copysync!(c,b)
        beta != 1 && scale!(beta, c)
        axpy!(alpha, a, c)
    end
    gpusync(); return c
end

function baddforw!(alpha::Number,a::Array,beta::Number,b::Array,c::Array)
    broadcast!(+,c,(alpha==1 ? a : alpha*a),(beta==1 ? b : beta*b))
end

@gpu function baddforw1!{T}(alpha::Number,a::CudaArray{T},beta::Number,b::CudaArray{T},c::CudaArray{T}) # mnist2d:6.40
    c===b || copysync!(c,b)
    cudnnAddTensor(a,c; alpha=T(alpha), beta=T(beta))
    gpusync(); return c
end

function cudnnAddTensorCompatible(a,b)
    sa = size(a)
    sb = size(b)
    sa == sb && return true
    n = length(sa)
    length(sb) == n || return false
    sa == ntuple(i->(i==n-1 ? sb[i] : 1), n) && return true
    sa == ntuple(i->(i==n ? 1 : sb[i]), n) && return true
    sa == ntuple(i->(i>=n-1 ? 1 : sb[i]), n) && return true
    return false
end

@gpu function baddforw2!{T}(alpha::Number,a::CudaArray{T},beta::Number,b::CudaArray{T},c::CudaArray{T}) # mnist2d:7.55
    for i=1:ndims(a) 
        size(a,i)==1 || size(a,i)==size(b,i) ||
        throw(DimensionMismatch("Each dimension of x1 must match x2 or be 1."))
    end
    ndims(b) <= 8 || error("add kernel supports dimensions up to 8")
    T <: Float32 ? ccall((:addforw32,libknet),Void,(Cint,Cfloat,Ptr{Cint},Ptr{Cfloat},Cfloat,Ptr{Cint},Ptr{Cfloat},Ptr{Cfloat}),ndims(a),T(alpha),cudadims(a),a,T(beta),cudadims(b),b,c) :
    T <: Float64 ? ccall((:addforw64,libknet),Void,(Cint,Cdouble,Ptr{Cint},Ptr{Cdouble},Cdouble,Ptr{Cint},Ptr{Cdouble},Ptr{Cdouble}),ndims(a),T(alpha),cudadims(a),a,T(beta),cudadims(b),b,c) :
    error("$T not supported")
    gpusync(); return c
end

# TODO: this does not cover all forms of db for cpu:
function baddback!(dy::Array, db::Vector)
    c=ndims(dy)-1
    fillsync!(db, zero(eltype(db)))
    @inbounds for i=1:length(dy)
        db[ind2sub(size(dy),i)[c]] += dy[i]
    end
    return db
end

function baddback!(dy, db)
    size(db) == size(dy) || (db = reshape_to_match(db,dy; CUDNN_ADD_SAME_C=true))
    if size(db) == size(dy)
        db === dy   || copysync!(db, dy)
    elseif cudnnConvolutionBackwardBiasCompatible(dy,db)
        baddback1!(dy, db)
    elseif ndims(dy) == 2
        baddback2!(dy, db)
    else
        baddback3!(dy, db)
    end
end

function cudnnConvolutionBackwardBiasCompatible(dy,db)
    ndims(dy)==ndims(db)==1 && return (size(dy)==size(db))
    size(db) == ntuple(i->(i==ndims(dy)-1 ? size(dy,i) : 1), ndims(dy))
end

@gpu function baddback1!{T}(dy::CudaArray{T}, db::CudaArray{T})
    cudnnConvolutionBackwardBias(dy, db)
    gpusync(); return db
end

@gpu function baddback2!{T}(dy::CudaArray{T}, db::CudaArray{T})
    ndims(db) == ndims(dy) || throw(DimensionMismatch())
    ndims(db)==2 || throw(DimensionMismatch())
    if size(db,1)==size(dy,1) && size(db,2)==1
        tmp = fillsync!(similar(dy, (size(dy,2),1)),1)
        A_mul_B!(db,dy,tmp)
        free(tmp)
    elseif size(db,2)==size(dy,2) && size(db,1)==1
        tmp = fillsync!(similar(dy, (1,size(dy,1))),1)
        A_mul_B!(db,tmp,dy)
        free(tmp)
    elseif size(db)==(1,1)
        baddback3!(dy,db)
    else
        throw(DimensionMismatch())
    end
    gpusync(); return db
end

@gpu function baddback3!{T}(dy::CudaArray{T}, db::CudaArray{T})
    ndims(db) == ndims(dy) || throw(DimensionMismatch())
    ndims(db) <= 8 || error("add kernel supports dimensions up to 8")
    for i=1:ndims(db) 
        size(db,i)==1 || size(db,i)==size(dy,i) ||
        throw(DimensionMismatch("Each dimension of x1 must match x2 or be 1."))
    end
    fillsync!(db,0)
    T <: Float32 ? ccall((:addback32,libknet),Void,(Cint,Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cfloat}),ndims(dy),cudadims(dy),dy,cudadims(db),db) :
    T <: Float64 ? ccall((:addback64,libknet),Void,(Cint,Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cdouble}),ndims(dy),cudadims(dy),dy,cudadims(db),db) :
    error("$T not supported")
    gpusync(); return db
end



"reshape_to_match(x1,x2) reshapes x1 to match x2."
function reshape_to_match(x1,x2; CUDNN_ADD_SAME_C=false)
    if CUDNN_ADD_SAME_C && ndims(x1)==1
        newsize = ntuple(ndims(x2)) do i
            i != ndims(x2)-1 ? 1 :
            size(x2,i) == size(x1,1) ? size(x1,1) :
            throw(DimensionMismatch("$(size(x1)) $(size(x2))"))
        end
    else
        newsize = ntuple(ndims(x2)) do i
            i > ndims(x1) ? 1 :
            size(x1,i)==1 ? 1 :
            size(x1,i)==size(x2,i) ? size(x1,i) :
            throw(DimensionMismatch("$(size(x1)) $(size(x2))"))
        end
    end
    reshape(x1, newsize)
end


function infersize(a::Add, x1, x2, y)
    if x1==x2==y==nothing
        nothing
    elseif x2==y==nothing
        (x1, nothing, nothing)
    elseif x2==nothing
        infersize(a, x1, y, y)
    elseif y==nothing
        infersize(a, x1, x2, x2)
    else
        ydims = map(x2, y) do xdim, ydim
            xdim == ydim ? xdim :
            xdim == 0 ? ydim :
            ydim == 0 ? xdim :
            throw(DimensionMismatch("$y=add($x1,$x2)"))
        end
        x1==nothing && (return (ydims, ydims, ydims))
        xdims = [x1...]; ydims = [ydims...]
        if length(xdims) == 1
            xdims[1] == ydims[end-1] ? nothing :
            xdims[1] == 0 ? xdims[1]=ydims[end-1] :
            ydims[end-1]==0 ? ydims[end-1]=xdims[1] :
            throw(DimensionMismatch("$y=add($x1,$x2)"))
        elseif length(x1) <= length(ydims)
            for i=1:length(xdims)
                xdims[i] == ydims[i] ? continue :
                xdims[i] == 1 ? continue :
                xdims[i] == 0 ? continue :
                ydims[i] == 0 ? (ydims[i] = xdims[i]) :
                throw(DimensionMismatch("$y=add($x1,$x2)"))
            end
        else
            throw(DimensionMismatch("$y=add($x1,$x2)"))
        end
        xdims = tuple(xdims...); ydims = tuple(ydims...)
        return (xdims, ydims, ydims)
    end
end



### DEAD CODE:

# biasforw(b::Vector, x::Array, y::Array)=(c=ndims(x)-1; for i=1:length(y); y[i] = x[i] + b[ind2sub(size(x),i)[c]]; end; y)
# biasforw(b::Vector, x::Vector, y::Vector)=(for i=1:length(y); y[i] = x[i] + b[i]; end; y)
# @gpu biasforw(b::CudaArray, x::CudaArray, y::CudaArray)=(y===x||copysync!(y,x);cudnnAddTensor(b, y; mode=CUDNN_ADD_SAME_C); gpusync(); y)

# function sizeafterbias(x1,x2)
#     i1 = x1[1]
#     i2 = x2[end-1]
#     i1 == 0 && (i1=i2)
#     i2 == 0 && (i2=i1)
#     i1 == i2 || error()
#     tuple(x2[1:end-2]..., i2, x2[end])
# end

# function commonsize(x1,x2)
#     map(x1, x2) do i1,i2
#         i1 == 0 && (i1=i2)
#         i2 == 0 && (i2=i1)
#         i1 == i2 || error()
#         i1
#     end
# end

    # elseif biassize(dy, dx1)
    #     biasback(dy, dx1)
    #     a.alpha == 1 || scale!(a.alpha, dx1)
    # else
    #     # TODO: implement more broadcasting back.
    #     error("Don't know how to do back pass with $(size(dy))=$(size(dx1))+$(size(dx2))")
# biassize(dy,db)=(size(db,1)==size(dy, ndims(dy)==1 ? 1 : ndims(dy)-1) && all([size(db,i)==1 for i=2:ndims(db)]))

# baddback!(alpha::Number, dy::Array, db::Vector)=(c=ndims(dy)-1; fillsync!(db, zero(eltype(db))); for i=1:length(dy); db[ind2sub(size(dy),i)[c]] += dy[i]; end; alpha==1||scale!(alpha,db))
# baddback!(alpha::Number, dy::Vector, db::Vector)=(for i=1:length(dy); db[i]=dy[i]; end; alpha==1||scale!(alpha,db))

# TODO: averaging? keep both arr and avg inside par, set output to one without copy?
# should be handled between net and par not by the op.
# DONE: handle nothings?  -- net is handling them.
# DONE: handle scalar input for adding a constant. -- axpb will handle this
# DONE: back

# add(x1,x2,y; alpha=1, beta=1)=(Add(alpha,beta),x1,x2,y)
# _KENV[:add] = Add
# _KENV[:+] = Add
