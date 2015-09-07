type Add2 <: Layer; Add2()=new(); dx1; dx2; end

ninputs(::Add2)=2
overwrites(l::Add2)=true
back_reads_x(l::Add2)=false
back_reads_y(l::Add2)=false

# x1, x2 are similarly sized input matrices, y=x1+x2

function forw(l::Add2, x1, x2; y=nothing, o...)
    y == nothing && (y=x2)      # we'll overwrite the second matrix for the result by default
    (issimilar(y,x1) && issimilar(y,x2)) || error("Input mismatch")
    (y===x2 ? axpy!(1,x1,y) :
     y===x1 ? axpy!(1,x2,y) :
     (copy!(y,x2); axpy!(1,x1,y)))
end

# if any of the input matrices is nothing, it represents the zero matrix, and we return the other matrix

forw(l::Add2, ::Void, ::Void; o...)=nothing
forw(l::Add2, x1, ::Void; y=nothing, o...)=(y==nothing ? x1 : copy!(y,x1))
forw(l::Add2, ::Void, x2; y=nothing, o...)=(y==nothing ? x2 : copy!(y,x2))


function back(l::Add2, dy; dx=nothing, returndx=true, o...)
    returndx || return
    dx == nothing || ((l.dx1,l.dx2) = dx)
    l.dx1 = (!isdefined(l,:dx1) ? copy(dy) :
             l.dx1 === dy ? dy :
             copy!(l.dx1, dy))
    l.dx2 = (!isdefined(l,:dx2) ? dy :
             l.dx2 === dy ? dy :
             copy!(l.dx2, dy))
    return (l.dx1,l.dx2)
end

# We want to pass back two copies of dy, but if we pass the same
# copy and one gets overwritten the other will be corrupted.  This
# may happen, for example, if one of our inputs has multiple
# outputs, so its dy is updated incrementally.  So we need
# separate copies.  In other words return (dy,dy) won't work.  We
# can pass back dy as one of the copies, we will allocate l.dx for
# the other.
