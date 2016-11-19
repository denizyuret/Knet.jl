using Knet: gpu

cudaProperties = [
    ("cudaComputeMajor","cudaComputeMajor","major"),
    ("cudaComputeMinor","cudaComputeMinor","minor"),
]

function cudapdef(f, j=f, o...)
    J=Symbol(j)
    @eval begin
        function $J(id::Int)
            ccall(($f,$libknet8),Cint,(Cint,),id)
        end
    end
end

for f in cudaProperties
    isa(f,Tuple) || (f=(f,))
    cudapdef(f...)
end

cudaComputeCapability() = string(cudaComputeMajor(gpu()), ".", cudaComputeMinor(gpu()))