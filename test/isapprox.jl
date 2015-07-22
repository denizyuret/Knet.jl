import Base: isapprox

function isapprox(x, y; 
                  maxeps::Real = max(eps(eltype(x)), eps(eltype(y))),
                  rtol::Real=maxeps^(1/4), atol::Real=maxeps^(1/2))
    size(x) == size(y) || (warn("isapprox: $(size(x))!=$(size(y))"); return false)
    isa(x, KUdense) && (x = x.arr)
    isa(y, KUdense) && (y = y.arr)
    KUnet.GPU && isa(x, AbstractCudaArray) && (x = to_host(x))
    KUnet.GPU && isa(y, AbstractCudaArray) && (y = to_host(y))
    d = abs(x-y)
    s = abs(x)+abs(y)
    all(d .< (atol + rtol * s))
end

