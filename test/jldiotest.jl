using Compat
using KUnet
using HDF5,JLD
typealias LUP Union(Op,UpdateParam)
import Base.isequal

function isequal(l1::LUP, l2::LUP)
    for n in fieldnames(l1)
        isdefined(l1,n) || continue
        isdefined(l2,n) || return false
    end
    for n in fieldnames(l2)
        isdefined(l2,n) || continue
        isdefined(l1,n) || return false
        isequal(l1.(n),l2.(n)) || return false
    end
    return true
end

net=newnet(relu, 1326, 20000,10)
setparam!(net, learningRate=0.02, dropout=0.5)
save("foo.jld", net)
foo=newnet("foo.jld")
isequal(copy(net,:cpu),copy(foo,:cpu))
