type Dot <: Op; Dot(;o...)=new(); end

ninputs(::Dot)=2
canoverwrite(::Dot)=false
back_reads_x(::Dot)=true
back_reads_y(::Dot)=false

function forw(::Dot, x1, x2, y; o...)
    (y === x1 || y === x2) && error("No array sharing in dot.")
    (x1 == nothing || x2 == nothing) && return nothing
    A_mul_B!(y, x1, x2)
end

function back(::Dot, dy, dx1, dx2; x=nothing, o...)
    if dx1 != nothing
        (dx1 === dy || dx1 === x[2]) && error("No array sharing in dot.")
        (x[2] != nothing ? A_mul_Bt!(dx1, dy, x[2]) : fillsync!(dx1, 0))
    end
    if dx2 != nothing
        (dx2 === dy || dx2 === x[1]) && error("No array sharing in dot.")
        (x[1] != nothing ? At_mul_B!(dx2, x[1], dy) : fillsync!(dx2, 0))
    end
end

function infersize(d::Dot,a,b,c)
    a==b==c==nothing && return nothing
    # a,b may have more than 2 dims (convolution tensors etc.), in which case we group the first n-1 into a super-column
    # or they could have 1 dim (bias), in which case we assume a column vector
    # a,b can also be nothing
    dotsize(x)=(length(x)==1 ? (x[1],1) : (prod(x[1:end-1]),x[end]))
    a==nothing || (a=[a...]; (a1,a2)=dotsize(a))
    b==nothing || (b=[b...]; (b1,b2)=dotsize(b))
    (c1,c2) = (c == nothing ? (0,0) :
               length(c)==2 ? c :
               throw(DimensionMismatch()))
    if a != nothing && b != nothing
        a2 == b1 ? nothing :
        a2 == 0  ? a2=b1 :
        b1 == 0  ? b1=a2 :
        throw(DimensionMismatch())
    end
    if a != nothing
        a1 == c1 ? nothing :
        a1 == 0  ? a1=c1 :
        c1 == 0  ? c1=a1 :
        throw(DimensionMismatch())
        length(a) <= 2 && (a[1] = a1)
        length(a) >= 2 && (a[end] = a2)
    end
    if b != nothing
        b2 == c2 ? nothing :
        b2 == 0  ? b2=c2 :
        c2 == 0  ? c2=b2 :
        throw(DimensionMismatch())
        length(b) <= 2 && (b[1] = b1)
        length(b) >= 2 && (b[end] = b2)
    end
    (a==nothing ? a : tuple(a...),
     b==nothing ? b : tuple(b...),
     (c1,c2))
end


### DEAD CODE:

# import Base: dot

# TODO: Who implements averaging?  -- net should
# TODO: handle or don't use nothings? -- don't use

# dot(w,x,y)=(Dot(),w,x,y)
