type Mul <: Op; Mul()=new(); end

# TODO: implement broadcasting
"@knet function mul(x,y) is element-wise multiplication."
mul(x1,x2,y)=(Mul(),x1,x2,y)
ninputs(::Mul)=2
overwrites(::Mul)=false
back_reads_x(::Mul)=true
back_reads_y(::Mul)=false

# x1,x2 is a pair of similarly sized input matrices

function forw(l::Mul, x1, x2, y; o...)
    @assert x2 == nothing || size(x2) == size(y)
    if x1==nothing || x2==nothing # nothing represents zero
        nothing
    elseif size(x1) == size(x2)
        mul2!(y,x1,x2)          # TODO: (minor) change order to x1,x2,y
    else
        error()
    end
end

function back(l::Mul, dy, dx1, dx2; x=nothing, o...)
    if dx2 != nothing
        @assert size(dx2) == size(dy)
        if x[1] == nothing      # representing zero
            fill!(dx2, 0)
        elseif size(x[1]) == size(dy)
            mul2!(dx2, dy, x[1])
        else
            throw(DimensionMismatch())
        end
    end
    if dx1 == nothing
        # done
    elseif size(dx1) == size(dy)
        if x[2] == nothing      # representing zero
            fill!(dx1, 0)
        elseif size(x[2]) == size(dy)
            mul2!(dx1, dy, x[2])
        else
            error("x2 and y should have the same size in mul")
        end
    else
        throw(DimensionMismatch())
    end
end

function infersize(::Mul, x1, x2)
    if x1==x2==nothing
        nothing
    elseif x1==nothing
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

