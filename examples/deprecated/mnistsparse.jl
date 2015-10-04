# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# Testing sparse arrays.

using Base.Test
using Knet
isdefined(:MNIST) || include("mnist.jl")
include("mlp.jl")
atol2 = 0.01
rtol2 = 0.01
isapprox2(x,y)=isapprox(x.out,y.out;atol=atol2,rtol=rtol2)&&isapprox(x.dif,y.dif;atol=atol2,rtol=rtol2)
adense(x)=x

function mnistsparse(args=ARGS)
    info("Testing MNIST mlp with sparse arrays.")
    nbatch=100

    gcheck = 10
    net = cell(4)
    dtrn = cell(4)
    dtst = cell(4)
    iter = 0
    prog = mlp(layers=(64,10), loss=softmax, actf=relu, winit=Gaussian(0,.01), binit=Constant(0))
    results = Any[]

    for (fx,fy) in ((adense,adense), (adense,sparse), (sparse,adense), (sparse,sparse))
        iter += 1
        net[iter] = Net(prog)
        setopt!(net[iter], lr=0.5)
        dtrn[iter] = ItemTensor(fx(MNIST.xtrn), fy(MNIST.ytrn); batch=nbatch)
        dtst[iter] = ItemTensor(fx(MNIST.xtst), fy(MNIST.ytst); batch=nbatch)
        setseed(42)
        l=w=g=ltrn=atrn=ltst=atst=0
        for epoch=1:3
            # TODO: gcheck does not work ?
            (l,w,g) = train(net[iter], dtrn[iter]; gclip=0, gcheck=gcheck, getloss=true, getnorm=true, atol=atol2, rtol=rtol2)
            (ltrn,atrn,ltst,atst) = (test(net[iter], dtrn[iter]), 
                                     accuracy(net[iter], dtrn[iter]), 
                                     test(net[iter], dtst[iter]), 
                                     accuracy(net[iter], dtst[iter]))
        end
        @show (fx,fy,l,w,g,ltrn,atrn,ltst,atst)
        @show map(isapprox2, params(net[1]), params(net[iter]))
        push!(results, (fx,fy,l,w,g,ltrn,atrn,ltst,atst))
    end
    return results
end

!isinteractive() && !isdefined(:load_only) && mnistsparse(ARGS)


# Reference output for debugging.  The last two results with sparse input are not stable.

# [dy_052@hpc3001 examples]$ julia mnistsparse.jl 
# INFO: Loading MNIST...
#   5.478525 seconds (366.45 k allocations: 503.265 MB, 1.74% gc time)
#   8.876875 seconds (9.64 M allocations: 427.910 MB, 2.03% gc time)
# (fx,fy,l,w,g,ltrn,atrn,ltst,atst) = (adense,adense,0.10628127f0,24.865437f0,3.5134742f0,0.100041345f0,0.9681833333333333,0.114785746f0,0.9641)
# map(isapprox1,params(net[1]),params(net[iter])) = Bool[true,true,true,true]
#   8.316890 seconds (5.36 M allocations: 251.544 MB, 0.93% gc time)
# (fx,fy,l,w,g,ltrn,atrn,ltst,atst) = (adense,sparse,0.1062698f0,24.866688f0,3.5134742f0,0.100718f0,0.9679333333333333,0.1149149f0,0.9642)
# map(isapprox1,params(net[1]),params(net[iter])) = Bool[true,true,true,true]
#  12.682341 seconds (5.63 M allocations: 629.662 MB, 1.61% gc time)
# (fx,fy,l,w,g,ltrn,atrn,ltst,atst) = (sparse,adense,0.10628127f0,24.865438f0,3.5134711f0,0.100041375f0,0.9681833333333333,0.11478577f0,0.9641)
# map(isapprox1,params(net[1]),params(net[iter])) = Bool[true,true,true,true]
#  14.437739 seconds (5.15 M allocations: 622.468 MB, 1.36% gc time)
# (fx,fy,l,w,g,ltrn,atrn,ltst,atst) = (sparse,sparse,0.106173664f0,24.870913f0,3.5582156f0,0.10027795f0,0.9682,0.11406385f0,0.9646)
# map(isapprox1,params(net[1]),params(net[iter])) = Bool[false,false,false,true]

# Here is another run showing the unstability:

# [dy_052@hpc3001 examples]$ julia mnistsparse.jl 
# INFO: Loading MNIST...
#   5.480183 seconds (366.45 k allocations: 503.265 MB, 1.72% gc time)
#   8.797816 seconds (9.64 M allocations: 427.910 MB, 2.05% gc time)
# (fx,fy,l,w,g,ltrn,atrn,ltst,atst) = (adense,adense,0.10628127f0,24.865437f0,3.5134742f0,0.100041345f0,0.9681833333333333,0.114785746f0,0.9641)
# map(isapprox1,params(net[1]),params(net[iter])) = Bool[true,true,true,true]
#   7.973043 seconds (5.35 M allocations: 251.381 MB, 0.88% gc time)
# (fx,fy,l,w,g,ltrn,atrn,ltst,atst) = (adense,sparse,0.1062698f0,24.866688f0,3.5134742f0,0.100718f0,0.9679333333333333,0.1149149f0,0.9642)
# map(isapprox1,params(net[1]),params(net[iter])) = Bool[true,true,true,true]
#  11.925001 seconds (5.64 M allocations: 629.857 MB, 1.22% gc time)
# (fx,fy,l,w,g,ltrn,atrn,ltst,atst) = (sparse,adense,0.1061508f0,24.87466f0,3.5049448f0,0.09875954f0,0.9686833333333333,0.11317172f0,0.9654)
# map(isapprox1,params(net[1]),params(net[iter])) = Bool[false,true,false,true]
#  14.341379 seconds (5.15 M allocations: 621.469 MB, 1.00% gc time)
# (fx,fy,l,w,g,ltrn,atrn,ltst,atst) = (sparse,sparse,0.10627592f0,24.866186f0,3.5134723f0,0.09992511f0,0.9681666666666666,0.114817984f0,0.9641)
# map(isapprox1,params(net[1]),params(net[iter])) = Bool[true,true,true,true]
