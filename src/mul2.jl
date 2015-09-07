type Mul2 <: Layer; y; x1; x2; dx1; dx2; Mul2()=new(); end

ninputs(::Mul2)=2
overwrites(l::Mul2)=false
back_reads_x(l::Mul2)=true
back_reads_y(l::Mul2)=false

# x1,x2 is a pair of similarly sized input matrices

function forw(l::Mul2, x1, x2; y=nothing, o...)
    initforw(l,x1,x2,y)
    mul2!(l.y, x1, x2)
end

# if any of the matrices is nothing, that represents zero, we return nothing

forw(l::Mul2, ::Void, ::Void; o...)=nothing
forw(l::Mul2, x1, ::Void; o...)=nothing
forw(l::Mul2, ::Void, x2; o...)=nothing

function initforw(l::Mul2, x1, x2, y)
    issimilar(x1,x2) || error("Input mismatch")
    (l.x1,l.x2) = (x1,x2)
    y != nothing && (l.y = y)
    similar!(l, :y, x1)
    return l.y
end

function back(l::Mul2, dy; dx=nothing, x=(l.x1,l.x2), returndx=true, o...)
    (length(x)==2 && issimilar(dy,x[1]) && issimilar(dy,x[2])) || error("Input mismatch")
    returndx || return
    dx != nothing && ((l.dx1,l.dx2) = dx)
    similar!(l,:dx1,x[1])
    similar!(l,:dx2,x[2])
    mul2(l.dx1,dy,x[2])
    mul2(l.dx2,dy,x[1])
    return (l.dx1,l.dx2)
end
