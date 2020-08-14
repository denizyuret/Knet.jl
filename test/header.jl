using Test, Random, AutoGrad, Knet, CUDA
if isdefined(AutoGrad, :gradcheck); @eval begin
    using AutoGrad: gradcheck
end; end
if isdefined(AutoGrad, :gcheck); @eval begin
    using AutoGrad: gcheck, @gcheck
end; end
