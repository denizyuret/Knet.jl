type Mul2 <: Layer; y; x1; x2; dx1; dx2; Mul2()=new(); end

ninputs(::Mul2)=2
overwrites(l::Mul2)=false
back_reads_x(l::Mul2)=true
back_reads_y(l::Mul2)=false

# x is a pair (Tuple{2}) of similarly sized input matrices
# we'll overwrite the second matrix for the result for now
# if any of the matrices is nothing, that represents zero, we return nothing (or mult the other by zero?)
# TODO: We should probably implement dropout (and gaussian noise etc.) using mul2 and random generators.

function forw(l::Mul2, x1, x2; y=nothing, o...)
    return ((x1 == nothing || x2 == nothing) ? nothing :
            mul2!(initforw(l,x1,x2,y), x1, x2))
end

function initforw(l::Mul2, x1, x2, y)
    issimilar(x1,x2) || error("Input mismatch")
    (l.x1,l.x2) = (x1,x2)
    y != nothing && (l.y = y)
    similar!(l, :y, x1)
    return l.y
end

function back(l::Mul2, dy; x=(l.x1,l.x2), returndx=true, o...)
    (length(x)==2 && issimilar(dy,x[1]) && issimilar(dy,x[2])) || error("Input mismatch")
    returndx || return
    similar!(l,:dx1,x[1])
    similar!(l,:dx2,x[2])
    mul2(l.dx1,dy,x[2])
    mul2(l.dx2,dy,x[1])
    return (l.dx1,l.dx2)
end
