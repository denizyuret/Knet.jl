type Bias <: Layer; b::Param; end
Bias(b::Array;a...)=Bias(Param(b;a...))
Bias(b::CudaArray;a...)=Bias(Param(b;a...))
Bias(d::Integer;a...)=Bias(Param(zeros(Float32,d);a...))

update(l::Bias)=update(l.b)

function forw(l::Bias, x; o...)
    @in1! x .+ l.b.data
    return x
end

function back(l::Bias, dy; o...)
    chksize(l.b, :diff, l.b.data)
    sum!(l.b.diff, dy)
    return dy
end

