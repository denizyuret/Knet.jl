abstract Broadcast <: Op
type Add <: Broadcast; Add(;o...)=new(); end
type Mul <: Broadcast; Mul(;o...)=new(); end

ninputs(::Broadcast)=2
canoverwrite(::Add)=true        # add can overwrite but only if output same size as one of the inputs
back_reads_x(::Add)=false
back_reads_y(::Add)=false
canoverwrite(::Mul)=false       # mul needs x for back, so cannot overwrite
back_reads_x(::Mul)=true
back_reads_y(::Mul)=false

# TODO: test this with 2d, 4d, lstm, irnn, minibatch=1 etc.
# better yet prove that it will work.
function infersize(::Broadcast, D...)
    findfirst(d->!isa(d,Void), D) == 0 && return nothing
    maxd = 0; for d in D
        d != nothing && length(d) > maxd && (maxd = length(d))
    end
    A = cell(length(D)); for i=1:length(D)
        A[i] = (D[i] == nothing ? zeros(Int,maxd) : Int[D[i]...])
    end
    length(A[end]) == maxd || throw(DimensionMismatch("$A"))
    for i=1:maxd
        maxi = 0; for a in A
            length(a) >= i && a[i] >= maxi && (maxi = a[i])
        end
        for a in A
            length(a) < i ? continue :
            a[i] == maxi ? continue :
            a[i] == 1 ? continue :
            a[i] == 0 ? a[i] = maxi :
            throw(DimensionMismatch("$A"))
        end
        A[end][i] == maxi || throw(DimensionMismatch("$A"))
    end
    ntuple(i->ntuple(j->A[i][j], length(A[i])), length(A))
end

### LEVEL 1: forw/back Handle `nothing`

function forw(::Add,x1,x2,y; o...)
    x1==x2==nothing ? nothing :         # `nothing` represents the zero array to avoid unnecessary fill! in forw and copy! in stack
    x1==nothing ? broadcast!(+,y,x2) :	# use broadcast here instead of copy in case x,y have different sizes
    x2==nothing ? broadcast!(+,y,x1) :
    broadcast!(+,y,x1,x2)               # forw uses julia broadcast semantics
end

function forw(::Mul,x1,x2,y;o...)
    x1==nothing ? nothing :             # `nothing` represents zero for mul also
    x2==nothing ? nothing :             # TODO: the alternative is to have it represent the ones array?
    broadcast!(*,y,x1,x2)               # but I had trouble with that before for a reason I don't remember.
end

function back(::Add,dy,dx1,dx2; o...)
    dx1!=nothing && addback(dy,dx1)
    dx2!=nothing && addback(dy,dx2)
end

function back(::Mul,dy,dx1,dx2; x=nothing, o...)
    (x==nothing || length(x)!=2) && error("back(mul) needs inputs x1 and x2")
    (x1,x2) = x
    dx1!=nothing && (x2==nothing ? fillsync!(dx1,0) : mulback(dy,x2,dx1))
    dx2!=nothing && (x1==nothing ? fillsync!(dx2,0) : mulback(dy,x1,dx2))
end

### LEVEL 2: broadcast!, add/mulback: select best op based on size

import Base: broadcast!, sum!

@gpu function broadcast!{T}(f::Function, y::CudaArray{T}, x::CudaArray{T})
    if f===+
        if size(y)===size(x)
            # Base.warn_once(:ADD2FORW)
            y===x || copysync!(y,x)  # addirnn=11.95 addlstm=2.48 copyseq=11.51 rnnlm=22.84
        else
            Base.warn_once(:BADD2FORW_NOT_TESTED)
            fill!(y,0)
            cudnnAddTensor(x,y)
            # baddforw(x,y)
        end
    else
        error("$f not supported")
    end
    return y
end

@gpu function broadcast!{T}(f::Function, y::CudaArray{T}, x1::CudaArray{T}, x2::CudaArray{T})
    if f===+
        if size(y)===size(x1)===size(x2)
            # Base.warn_once(:ADD3FORW) 
            # geam3(x1,x2,y)    # addirnn=12.18 addlstm=2.52 copyseq=11.59 rnnlm=22.97
            axpy3(x1,x2,y)      # addirnn=11.95 addlstm=2.48 copyseq=11.51 rnnlm=22.84
        else
            # Base.warn_once(:BADD3FORW) 
            # baddforw(x1,x2,y)      # mnist2d=4.27 4d=15.62 addirnn=14.75 addlstm=2.99 copyseq=15.39 rnnlm=22.28
            cudnnAddTensor3(x1,x2,y) # mnist2d=3.60 4d=14.42 addirnn=11.95 addlstm=2.48 copyseq=11.51 rnnlm=22.84
        end
    elseif f===*
        if size(y)===size(x1)===size(x2)
            # Base.warn_once(:MUL3FORW)
            mul3(x1,x2,y)       # addlstm=2.48 copyseq=11.51 rnnlm=22.84
        else
            Base.warn_once(:BMUL3FORW_NOT_TESTED)
            bmulforw(x1,x2,y)
        end
    else
        error("$f not supported")
    end
    return y
end

# addback sets dx=dy except if different sizes appropriate dims of dy are summed up
function addback(dy,dx)
    if dy === dx
        # done
    elseif size(dy) == size(dx)
        # Base.warn_once(:ADDBACK)
        copysync!(dx,dy)                    # addirnn=11.95 addlstm=2.48 copyseq=11.51 rnnlm=22.84
    else
        # Base.warn_once(:BADDBACK)
        sum!(dx,dy)
    end
end

# warning: this works for sum(a,2) but not for sum(a,1)
@gpu function sum!{T}(dx::CudaArray{T},dy::CudaArray{T})
    # baddback(dy,dx)                   # mnist2d=3.70 4d=22.36 addirnn=14.35 addlstm=2.90 copyseq=11.32 rnnlm=20.94   nondeterministic.
    cudnnConvolutionBackwardBias(dy,dx) # mnist2d=3.60 4d=14.42 addirnn=11.95 addlstm=2.48 copyseq=11.51 rnnlm=22.84
end

# mulback has dx1=dy*x2 and dx2=dy*x1 with appropriate broadcasting sums
function mulback(dy,x1,dx2)
    if size(dy)==size(x1)==size(dx2)
        # Base.warn_once(:MULBACK)
        mul3(dy,x1,dx2)         # addlstm=2.48 copyseq=11.51 rnnlm=22.84
    else
        Base.warn_once(:BMULBACK_NOT_TESTED)
        bmulback(dy,x1,dx2)
    end
end

### LEVEL 3: Actual implementations

# 3a. same size add

function axpy3(a,b,c)
    size(a)==size(b)==size(c) || throw(DimensionMismatch())
    c===a ? axpy!(1,b,c) :
    c===b ? axpy!(1,a,c) :
    (copysync!(c,b); axpy!(1,a,c))
    gpusync(); return c
end

@gpu function geam3{T}(a::CudaArray{T},b::CudaArray{T},c::CudaArray{T})
    size(a)==size(b)==size(c) || throw(DimensionMismatch())
    CUBLAS.geam!('N','N',T(1),mat2d(a),T(1),mat2d(b),mat2d(c))
    gpusync(); return c
end

# 3b. same size mul

@gpu function mul3{T}(a::CudaArray{T},b::CudaArray{T},c::CudaArray{T})
    size(a)==size(b)==size(c) || throw(DimensionMismatch())
    T <: Float32 ? ccall((:mul32,libknet),Void,(Cint,Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}), length(c),a,b,c) :
    T <: Float64 ? ccall((:mul64,libknet),Void,(Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),length(c),a,b,c) :
    error("$T not supported")
    gpusync(); return c
end

function mul3{T}(a::Array{T},b::Array{T},c::Array{T})
    size(a)==size(b)==size(c) || throw(DimensionMismatch())
    @inbounds for i=1:length(c); c[i]=a[i]*b[i]; end
    return c
end

# 3b. broadcasting add

@gpu function cudnnAddTensor3{T}(a::CudaArray{T},b::CudaArray{T},c::CudaArray{T})
    c===a ? cudnnAddTensor(b,c) :
    c===b ? cudnnAddTensor(a,c) :
    size(c)==size(b) ? (copysync!(c,b); cudnnAddTensor(a,c)) :
    size(c)==size(a) ? (copysync!(c,a); cudnnAddTensor(b,c)) :
    throw(DimensionMismatch())
    gpusync(); return c
end

# At least avoid alloc for cudadims (still paying copy cost)
_x1dims = CudaArray(Cint,8)
_x2dims = CudaArray(Cint,8)
_ydims = CudaArray(Cint,8)

@gpu function baddforw{T}(x::CudaArray{T}, y::CudaArray{T})
    ndims(x) <= ndims(y) <= 8 || error("Only xdims<=ydims<=8 supported")
    cudadims!(_x1dims, size(x), size(y))
    cudadims!(_ydims, size(y))
    T <: Float32 ? ccall((:badd2forw32,libknet),Void,(Cint,Cint,Ptr{Cint},Ptr{Cfloat}, Ptr{Cint},Ptr{Cfloat}), length(y),ndims(y),_x1dims,x,_ydims,y) :
    T <: Float64 ? ccall((:badd2forw64,libknet),Void,(Cint,Cint,Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cdouble}),length(y),ndims(y),_x1dims,x,_ydims,y) :
    error("$T not supported")
    gpusync(); return y
end

@gpu function baddforw{T}(x1::CudaArray{T}, x2::CudaArray{T}, y::CudaArray{T})
    (ndims(x1) <= ndims(y) && ndims(x2) <= ndims(y) && ndims(y) <= 8) || error("Only xdims<=ydims<=8 supported")
    cudadims!(_x1dims, size(x1), size(y))
    cudadims!(_x2dims, size(x2), size(y))
    cudadims!(_ydims, size(y))
    T <: Float32 ? ccall((:badd3forw32,libknet),Void,(Cint,Cint,Ptr{Cint},Ptr{Cfloat}, Ptr{Cint},Ptr{Cfloat}, Ptr{Cint},Ptr{Cfloat}), length(y),ndims(y),_x1dims,x1,_x2dims,x2,_ydims,y) :
    T <: Float64 ? ccall((:badd3forw64,libknet),Void,(Cint,Cint,Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cdouble}),length(y),ndims(y),_x1dims,x1,_x2dims,x2,_ydims,y) :
    error("$T not supported")
    gpusync(); return y
end

# 3c. broadcasting add back

@gpu function baddback{T}(dy::CudaArray{T}, dx::CudaArray{T})
    ndims(dx) <= ndims(dy) <= 8 || error("Only _x1dims<=ydims<=8 supported")
    cudadims!(_x1dims, size(dx), size(dy))
    cudadims!(_ydims, size(dy))
    fill!(dx,0)
    T <: Float32 ? ccall((:badd2back32,libknet),Void,(Cint,Cint,Ptr{Cint},Ptr{Cfloat}, Ptr{Cint},Ptr{Cfloat}), length(dy),ndims(dy),_ydims,dy,_x1dims,dx) :
    T <: Float64 ? ccall((:badd2back64,libknet),Void,(Cint,Cint,Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cdouble}),length(dy),ndims(dy),_ydims,dy,_x1dims,dx) :
    error("$T not supported")
    gpusync();
end


@gpu function bmulback{T}(dy::CudaArray{T}, x1::CudaArray{T}, dx2::CudaArray{T})
    error(:BMULBACK_NOT_IMPLEMENTED)
    ndims(dx) <= ndims(dy) <= 8 || error("Only xdims<=ydims<=8 supported")
    cudadims!(_x1dims, size(dx), size(dy))
    cudadims!(_ydims, size(dy))
    fill!(dx,0)
    T <: Float32 ? ccall((:bmul3back32,libknet),Void,(Cint,Cint,Ptr{Cint},Ptr{Cfloat}, Ptr{Cint},Ptr{Cfloat}), length(dy),ndims(dy),_ydims,dy,_x1dims,dx) :
    T <: Float64 ? ccall((:bmul3back64,libknet),Void,(Cint,Cint,Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cdouble}),length(dy),ndims(dy),_ydims,dy,_x1dims,dx) :
    error("$T not supported")
    gpusync();
end

@gpu function cudadims!(c::CudaArray{Cint}, xdims::Dims, ydims::Dims)
    d = ones(Cint, length(ydims))
    @inbounds for i=1:length(xdims)
        d[i] = xdims[i]
        d[i] == 1 || d[i] == ydims[i] || throw(DimensionMismatch("$xdims,$ydims"))
    end
    copysync!(c,1,d,1,length(d))
end

@gpu cudadims!(c::CudaArray{Cint},ydims::Dims)=copysync!(c, 1, Cint[ydims...], 1, length(ydims))

