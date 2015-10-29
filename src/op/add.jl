# TODO: averaging? keep both arr and avg inside par, set output to one without copy?
# should be handled between net and par not by the op.
# DONE: handle nothings?  -- net is handling them.
# DONE: handle scalar input for adding a constant. -- axpb will handle this
# TODO: back

type Add <: Op; alpha; beta; end

"""

@knet function add(x1,x2; alpha=1,beta=1) performs element-wise
broadcasting addition.  alpha and beta are used to scale x1 and x2
respectively.  The result computed is y = alpha x1 + beta x2.  The
size of the output y always matches the size of x2.  Broadcasting is
performed as follows (following Julia convention, size below refers to
the tuple of dimensions, fastest changing first, and ndims refers to
the number of dimensions):

- If x1 and x2 are the same size their elements are added.

- If x1 and x2 have the same ndims but different sizes, each dimension
  of the x1 array must match the coresponding dimension of the x2
  array or must be equal to 1. In the latter case, the same value from
  the x1 array for those dimensions will be used to blend into the x2
  array.  (At least this is what it says in the cudnnAddTensor doc but
  unfortunately not all combinations work as promised. TODO: I will
  update this doc as soon as Nvidia tells me what the real spec is.)
  Example: (5,4,1,1)+(5,4,3,2)=>(5,4,3,2)

- If x1 and x2 have different ndims, x1 is assumed to be missing its
  rightmost dims and those are assumed to be 1.  Example:
  (5,4)+(5,4,3,2)=>(5,4,1,1)+(5,4,3,2)=>(5,4,3,2).

- The one exception to the last rule is when x1 has ndims=1 and x2 has
  ndims > 1.  In this case x1 is matched to the next to last dimension
  of x2.  Example: (3,)+(5,4,3,2)=>(1,1,3,1)+(5,4,3,2)=>(5,4,3,2).

"""
add(x1,x2,y; alpha=1, beta=1)=(Add(alpha,beta),x1,x2,y)

ninputs(::Add)=2
overwrites(::Add)=true
back_reads_x(::Add)=false
back_reads_y(::Add)=false

add!(a::Number,x::Array,b::Number,y::Array)=broadcast!(+,y,(a==1 ? x : a*x),(b==1 ? y : b*y))
@gpu add!(a::Number,x::CudaArray,b::Number,y::CudaArray)=(cudnnAddTensor(x,y; alpha=a, beta=b); gpusync(); y)

function forw(a::Add, x1, x2, y; o...)
    @assert x2 == nothing || size(y) == size(x2)
    if x1!=nothing && x2!=nothing # we use nothing to represent the zero array
        x1 = reshape_to_match(x1,x2)
        y===x2 ? add!(a.alpha,x1,a.beta,y) :
        y===x1 ? add!(a.beta,x2,a.alpha,y) :
        (copy!(y,x2); add!(a.alpha,x1,a.beta,y))
    elseif x2 != nothing
        y===x2 ? y : copy!(y, x2)
    elseif x1 != nothing
        size(y) != size(x1) ? nothing :
        y===x1 ? y : copy!(y, x1)
    else
        nothing
    end
end

function reshape_to_match(x1,x2)
    if ndims(x1) < ndims(x2)
        n = ndims(x2)
        newsize = ones(Int,n)
        if ndims(x1)==1
            newsize[n-1] = length(x1)
        else
            for i=1:ndims(x1)
                newsize[i] = size(x1,i)
            end
        end
        x1 = reshape(x1, tuple(newsize...))
    end
    for i=1:ndims(x1)
        size(x1,i)==1 || size(x1,i)==size(x2,i) ||
        throw(DimensionMismatch("Each x1 dimension must match x2 or be 1."))
    end
    return x1
end

function back(a::Add, dy, dx1, dx2; o...)
    if dx2 != nothing
        size(dx2) == size(dy) || throw(DimensionMismatch("The size of the output must match x2"))
        dx2 === dy  || copy!(dx2, dy)
        a.beta == 1 || scale!(a.beta, dx2)
    end
    if dx1 == nothing
        # done
    elseif size(dx1) == size(dy)
        dx1 === dy   || copy!(dx1, dy)
        a.alpha == 1 || scale!(a.alpha, dx1)
    elseif size(dx1) == biassize(dy)
        biasback(dy, dx1)
        a.alpha == 1 || scale!(a.alpha, dx1)
    else
        # TODO: implement more broadcasting back.
        error("Don't know how to do back pass with $(size(dy))=$(size(dx1))+$(size(dx2))")
    end
end

biassize(y)=(size(y, ndims(y)==1 ? 1 : ndims(y)-1),)
biasback(dy::Array, db::Vector)=(c=ndims(dy)-1; fill!(db, zero(eltype(db))); for i=1:length(dy); db[ind2sub(size(dy),i)[c]] += dy[i]; end)
biasback(dy::Vector, db::Vector)=(for i=1:length(dy); db[i]=dy[i]; end)
@gpu biasback(dy::CudaArray, db::CudaArray)=(cudnnConvolutionBackwardBias(dy, db); gpusync(); db)

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
            throw(DimensionMismatch())
        end
        x1==nothing && (return (nothing, ydims, ydims))
        xdims = [x1...]; ydims = [ydims...]
        if length(xdims) == 1
            xdims[1] == ydims[end-1] ? nothing :
            xdims[1] == 0 ? xdims[1]=ydims[end-1] :
            ydims[end-1]==0 ? ydims[end-1]=xdims[1] :
            throw(DimensionMismatch())
        elseif length(x1) <= length(ydims)
            for i=1:length(xdims)
                xdims[i] == ydims[i] ? continue :
                xdims[i] == 1 ? continue :
                xdims[i] == 0 ? continue :
                ydims[i] == 0 ? (ydims[i] = xdims[i]) :
                throw(DimensionMismatch())
            end
        else
            throw(DimensionMismatch())
        end
        xdims = tuple(xdims...); ydims = tuple(ydims...)
        return (xdims, ydims, ydims)
    end
end



### DEAD CODE:

# biasforw(b::Vector, x::Array, y::Array)=(c=ndims(x)-1; for i=1:length(y); y[i] = x[i] + b[ind2sub(size(x),i)[c]]; end; y)
# biasforw(b::Vector, x::Vector, y::Vector)=(for i=1:length(y); y[i] = x[i] + b[i]; end; y)
# @gpu biasforw(b::CudaArray, x::CudaArray, y::CudaArray)=(y===x||copy!(y,x);cudnnAddTensor(b, y; mode=CUDNN_ADD_SAME_C); gpusync(); y)

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
