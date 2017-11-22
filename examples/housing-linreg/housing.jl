for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include(Pkg.dir("Knet","data","housing.jl"))

"""
This example uses the
[Boston Housing](https://archive.ics.uci.edu/ml/machine-learning-databases/housing) dataset
from the UCI Machine Learning Repository to demonstrate a linear
regression model. The dataset has housing related information for 506
neighborhoods in Boston from 1978. Each neighborhood has 14
attributes, the goal is to use the first 13, such as average number of
rooms per house, or distance to employment centers, to predict the
14’th attribute: median dollar value of the houses.

You can run the demo using `julia housing.jl`.  Use `julia housing.jl
--help` for a list of options.  The dataset will be automatically
downloaded and randomly split into training and test sets.  The
quadratic loss for the training and test sets will be printed at every
epoch and optimized parameters will be returned.

"""
module Housing
using Knet,ArgParse

predict(w,x)=(w[1]*x.+w[2])

loss(w,x,y)=(sum(abs2,y-predict(w,x)) / size(x,2))

lossgradient = grad(loss)

function train(w, x, y; lr=.1, epochs=20)
    for epoch=1:epochs
        g = lossgradient(w, x, y)
        update!(w, g; lr=lr)
    end
    return w
end

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="housing.jl (c) Deniz Yuret, 2016. Linear regression model for the Housing dataset from the UCI Machine Learning
Repository."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--epochs"; arg_type=Int; default=20; help="number of epochs for training")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
        ("--test"; arg_type=Float64; default=0.0; help="ratio of data to split for testing")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients")
    end
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    println(s.description)
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    w = map(atype, [ 0.1*randn(1,13), 0.1*randn(1,1) ])
    (xtrn,ytrn,xtst,ytst) = map(atype, Main.housing(o[:test]))
    report(epoch)=println((:epoch,epoch,:trn,loss(w,xtrn,ytrn),:tst,loss(w,xtst,ytst)))
    if o[:fast]
        @time (train(w, xtrn, ytrn; lr=o[:lr], epochs=o[:epochs]); gpu()>=0 && Knet.cudaDeviceSynchronize())
    else
        report(0)
        @time for epoch=1:o[:epochs]
            train(w, xtrn, ytrn; lr=o[:lr], epochs=1)
            report(epoch)
            if o[:gcheck] > 0
                gradcheck(loss, w, xtst, ytst; gcheck=o[:gcheck], verbose=true)
            end
        end
    end
    return w
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia housing.jl --epochs 10
# julia> Housing.main("--epochs 10")
PROGRAM_FILE=="housing.jl" && main(ARGS)

end # module Housing

# SAMPLE RUN 65f57ff+ Wed Sep 14 10:02:30 EEST 2016
#
# housing.jl (c) Deniz Yuret, 2016. Linear regression model for the Housing dataset from the UCI Machine Learning
# Repository.
# opts=(:seed,-1)(:epochs,20)(:lr,0.1)(:atype,"KnetArray")(:gcheck,0)(:fast,true)
# size(data) = (14,506)
# (:epoch,0,:trn,577.8060720494859,:tst,656.5939685291422)
#   0.018364 seconds (7.15 k allocations: 347.266 KB)
# (:epoch,20,:trn,20.908099852265146,:tst,33.9047224887537)
