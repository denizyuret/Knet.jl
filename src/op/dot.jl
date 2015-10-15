import Base: dot

# TODO: Who implements averaging?  -- net should
# TODO: handle or don't use nothings? -- don't use

type Dot <: Op; end

dot(w,x,y)=(Dot(),w,x,y)
ninputs(::Dot)=2
overwrites(::Dot)=false
back_reads_x(::Dot)=true
back_reads_y(::Dot)=false

function forw(::Dot, x1, x2, y; o...)
    if x1 == nothing || x2 == nothing
        return nothing
    end
    A_mul_B!(y, x1, x2)
end

function back(::Dot, dy, dx1, dx2; x=nothing, o...)
    dx1 != nothing && (x[2] != nothing ? A_mul_Bt!(dx1, dy, x[2]) : fill!(dx1, 0))
    dx2 != nothing && (x[1] != nothing ? At_mul_B!(dx2, x[1], dy) : fill!(dx2, 0))
end

function infersize(::Dot,a,b)
    # a,b may have more than 2 dims (convolution tensors etc.),
    # in which case we group the first n-1 into a super-column
    # a,b can also be nothing
    if a==b==nothing
        return nothing
    elseif a==nothing
        return (a, b, (0,b[end]))
    elseif b==nothing
        return (a, b, (prod(a[1:end-1]),0))
    else
        a = [a...]
        b = [b...]
        ma = prod(a[1:end-1])
        na = a[end]
        mb = prod(b[1:end-1])
        nb = b[end]
        na == 0 && (na = a[end] = mb)
        mb == 0 && length(b) == 2 && (mb = b[1] = na)
        @assert na == 0 || mb == 0 || na == mb
        (tuple(a...), tuple(b...), (ma,nb))
    end
end
