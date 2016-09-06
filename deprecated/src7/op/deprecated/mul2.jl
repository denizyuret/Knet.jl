type Mul2 <: Op; ybuf; x1; x2; dx1; dx2; Mul2()=new(); end

params(::Mul2)=Any[]
ninputs(::Mul2)=2
ysize(::Mul2,x1,x2)=size(x1)
overwrites(::Mul2)=false
back_reads_x(::Mul2)=true
back_reads_y(::Mul2)=false

# x1,x2 is a pair of similarly sized input matrices

function forw(l::Mul2, x1, x2; y=nothing, o...)
    (l.x1,l.x2) = (x1,x2)
    y = initforw(l,x1,x2,y)
    mul2!(y,x1,x2)
end

function initforw(l::Mul2, x1, x2, y)
    issimilar(x1,x2) || error("Input mismatch")
    y == nothing && (y = similar!(l, :ybuf, x1))
    issimilar0(y,x1) || error("Input output mismatch")
    return y
end

function back(l::Mul2, dy; dx=nothing, x=(l.x1,l.x2), returndx=true, o...)
    returndx || return
    (dx1,dx2) = initback(l, dy, x, dx)
    mul2!(dx1,dy,x[2])
    mul2!(dx2,dy,x[1])
    return (dx1,dx2)
end

function initback(l::Mul2, dy, x, dx)
    length(x)==2 || error("Need two inputs")
    (x1,x2) = x
    issimilar0(x1,x2) || error("Input mismatch")
    issimilar0(x1,dy) || error("Input gradient mismatch")
    issimilar0(x2,dy) || error("Input gradient mismatch")
    dx == nothing && (dx = (similar!(l,:dx1,x1), similar!(l,:dx2,x2)))
    issimilar0(x1,dx[1]) || error("Gradient mismatch")
    issimilar0(x2,dx[2]) || error("Gradient mismatch")
    dx[1] === dx[2] && error("Need two different dx")
    return dx
end

# if any of the matrices is nothing, that represents zero, we return nothing

forw(l::Mul2, ::Void, ::Void; o...)=nothing
forw(l::Mul2, x1, ::Void; o...)=nothing
forw(l::Mul2, ::Void, x2; o...)=nothing

ysize(l::Mul2, ::Void, ::Void)=nothing
ysize(l::Mul2, x1, ::Void)=nothing
ysize(l::Mul2, ::Void, x2)=nothing

mul2!(c,a::Void,b::Void)=fill!(c,0)
mul2!(c,a,b::Void)=fill!(c,0)
mul2!(c,a::Void,b)=fill!(c,0)

issimilar0(a,b)=issimilar(a,b)
issimilar0(a::Void,b::Void)=true
issimilar0(a,b::Void)=true
issimilar0(a::Void,b)=true

similar!(l,n,::Void)=nothing    # don't get rid of existing storage but return nothing
