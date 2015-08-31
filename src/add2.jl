type Add2 <: Layer; Add2()=new(); dx; end

ninputs(::Add2)=2
overwrites(l::Add2)=true
back_reads_x(l::Add2)=false
back_reads_y(l::Add2)=false

# x is a pair (Tuple{2}) of similarly sized input matrices
# if any of the input matrices is nothing, it represents the zero matrix, and we return the other matrix
# we'll overwrite the second matrix for the result

forw(l::Add2, x1, x2; o...)=(x1 == nothing ? x2 :
                             x2 == nothing ? x1 :
                             axpy!(1, x1, x2))

function back(l::Add2, dy; o...)
    # We want to pass back two copies of dy, but if we pass the same
    # copy and one gets overwritten the other will be corrupted.  This
    # may happen, for example, if one of our inputs has multiple
    # outputs, so its dy is updated incrementally.  So we need
    # separate copies.  In other words return (dy,dy) won't work.  We
    # can pass back dy as one of the copies, we will allocate l.dx for
    # the other.
    isdefined(l,:dx) ? copy!(l.dx,dy) : (l.dx=copy(dy))
    return (dy,l.dx)
end
