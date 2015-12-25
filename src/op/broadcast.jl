abstract Broadcast <: Op
type Add <: Broadcast; Add(;,o...)=new(); end
type Mul <: Broadcast; Mul(;,o...)=new(); end

ninputs(::Broadcast)=2
canoverwrite(::Add)=true        # add can overwrite but only if output same size as one of the inputs
back_reads_x(::Add)=false
back_reads_y(::Add)=false
canoverwrite(::Mul)=false       # mul needs x for back, so cannot overwrite
back_reads_x(::Mul)=true
back_reads_y(::Mul)=false

function forw(a::Add,x1,x2,y; o...)
    x1==x2==nothing ? nothing :         # `nothing` represents the zero array to avoid unnecessary fill! in forw and copy! in stack
    x1==nothing ? broadcast!(+,y,x2) :	# use broadcast here instead of copy in case x,y have different sizes
    x2==nothing ? broadcast!(+,y,x1) :
    broadcast!(+,y,x1,x2)               # forw uses julia broadcast semantics
end

function forw(a::Mul,x1,x2,y;o...)
    x1==nothing ? nothing :             # `nothing` represents zero for mul also
    x2==nothing ? nothing :             # TODO: the alternative is to have it represent the ones array?
    broadcast!(*,y,x1,x2)               # but I had trouble with that before for a reason I don't remember.
end

function back(a::Add,dy,dx1,dx2; o...)
    dx1!=nothing && addback(dy,dx1)
    dx2!=nothing && addback(dy,dx2)
end

function back(a::Mul,dy,dx1,dx2; x=nothing, o...)
    (x==nothing || length(x)!=2) && error("back(mul) needs inputs x1 and x2")
    (x1,x2) = x
    dx1!=nothing && (x2==nothing ? fillsync!(dx1,0) : mulback(x2,dy,dx1))
    dx2!=nothing && (x1==nothing ? fillsync!(dx2,0) : mulback(x1,dy,dx2))
end

@gpu function broadcast!{T}(f::Function, y::CudaArray{T}, x::CudaArray{T})
    if f===+
        if size(y)===size(x)
            y===x || copysync!(y,x)
        else
            error("TODO1")
        end
    else
        error("$f not supported")
    end
    gpusync(); return y
end

@gpu function broadcast!{T}(f::Function, y::CudaArray{T}, x1::CudaArray{T}, x2::CudaArray{T})
    if f===+
        if size(y)===size(x1)===size(x2)
            add_axpy(x1,x2,y)
        else
            error("TODO2")
        end
    elseif f===*
        if size(y)===size(x1)===size(x2)
            error("TODO3")
        else
            error("TODO4")
        end
    else
        error("$f not supported")
    end
end

# addback sets dx=dy except if different sizes appropriate dims of dy are summed up
function addback(dy,dx)
    error("TODO5")
end

# mulback has dx1=dy*x2 and dx2=dy*x1 with appropriate broadcasting sums
function mulback(x,dy,dx)
    error("TODO6")
end

### same size addforw:
# defined multiple functions for speed comparison, axpy seems fastest.

function add_axpy(a,b,c)
    c===a ? axpy!(1,b,c) :
    c===b ? axpy!(1,a,c) :
    (copysync!(c,b); axpy!(1,a,c))
    gpusync(); return c
end

@gpu function add_cudnn{T}(a::CudaArray{T},b::CudaArray{T},c::CudaArray{T})
    c===a ? CUDNN.cudnnAddTensor(b,c) :
    c===b ? CUDNN.cudnnAddTensor(a,c) :
    (copysync!(c,b); CUDNN.cudnnAddTensor(a,c))
    gpusync(); return c
end

@gpu function add_geam{T}(a::CudaArray{T},b::CudaArray{T},c::CudaArray{T})
    CUBLAS.geam!('N','N',T(1),a,T(1),b,c)
    gpusync(); return c
end

