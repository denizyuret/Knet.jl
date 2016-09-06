type Add2 <: Op; Add2()=new(); dx1; dx2; end

params(::Add2)=Any[]
ninputs(::Add2)=2
ysize(::Add2,x1,x2)=size(x1)
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
forw(l::Add2, x1, ::Void; y=nothing, o...)=(y==nothing ? x1 : y===x1 ? x1 : copy!(y,x1))
forw(l::Add2, ::Void, x2)=forw(l,x2,nothing)
ysize(l::Add2, ::Void, ::Void)=nothing
ysize(l::Add2, ::Void, x2)=ysize(l,x2,nothing)

function back(l::Add2, dy; dx=nothing, returndx=true, o...)
    returndx || return
    dx == nothing && (dx = (similar!(l,:dx1,dy), similar!(l,:dx2,dy)))
    dx[1] === dy || copy!(dx[1], dy)
    dx[2] === dy || copy!(dx[2], dy)
    return dx
end

# We want to pass back two copies of dy, but if we pass the same copy
# and one gets overwritten the other will be corrupted.  This may
# happen, for example, if one of our inputs has multiple outputs, so
# its dy is updated incrementally.  Or if one input is an overwriting
# layer it will overwrite the one copy with its dx.  So we may need
# separate copies.  In other words return (dy,dy) won't always work.
