type Mul <: Op; Mul()=new(); end

mul()=Mul()
ninputs(::Mul)=2
overwrites(::Mul)=false
back_reads_x(::Mul)=true
back_reads_y(::Mul)=false

# x1,x2 is a pair of similarly sized input matrices

function forw(l::Mul, x1, x2, y; o...)
    @assert size(x2) == size(y)
    if length(x1) == 1
        y === x2 || copy!(y,x2)
        scale!(x1[1], y)
    elseif size(x1) == size(x2)
        mul2!(y,x1,x2)          # TODO: (minor) change order to x1,x2,y
    else
        error()
    end
end

function back(l::Mul, dy, dx1, dx2; x=nothing, o...)
    if dx2 != nothing
        @assert size(dx2) == size(dy) == size(x[1])
        mul2!(dx2, dy, x[1])
    end
    if dx1 == nothing
        # done
    elseif size(dx1) == size(dy)
        @assert size(x[2]) == size(dy)
        mul2!(dx1, dy, x[2])
    elseif length(dx1) == 1
        error("not implemented") # TODO
    else
        error()
    end
end

function infersize(::Mul, x1, x2)
    if x1==x2==nothing
        nothing
    elseif x1==nothing
        (x1,x2,x2)
    elseif length(x1) == 1 && x1[1] == 1  # scalar mul
        (x1,x2,x2)
    elseif x2==nothing          # element-wise mul
        (x1, x1, x1)
    elseif length(x1) == length(x2)
        x3 = map(x1, x2) do i1,i2
            i1 == 0 && (i1=i2)
            i2 == 0 && (i2=i1)
            i1 == i2 || error()
            i1
        end
        (x3, x3, x3)
    else
        error()
    end
end

