type Bias <: Layer; b::Param; end
Bias(b::Array;a...)=Bias(Param(b;a...))
Bias(b::CudaArray;a...)=Bias(Param(b;a...))
# TODO: get rid of Float32
Bias(d::Integer;a...)=Bias(Param(zeros(Float32,d);a...))

update(l::Bias)=update(l.b)
setparam!(l::Bias,k,v)=setparam!(l.b,k,v)

forw(l::Bias, x::CudaArray; o...)=(cudnnAddTensor(l.b.data, x); x)
back(l::Bias, dy::CudaArray; o...)=(chksize(l.b, :diff, l.b.data); cudnnConvolutionBackwardBias(dy, l.b.diff); dy)

# TODO: make sure N-D works on cpu
function forw(l::Bias, x; o...)
    @in1! x .+ l.b.data
    return x
end

function back(l::Bias, dy; o...)
    chksize(l.b, :diff, l.b.data)
    sum!(l.b.diff, dy)
    return dy
end

