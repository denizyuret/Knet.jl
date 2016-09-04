"""
This example uses the Housing dataset from the UCI Machine Learning
Repository to demonstrate a linear regression model. The dataset has
housing related information for 506 neighborhoods in Boston from
1978. Each neighborhood has 14 attributes, the goal is to use the
first 13, such as average number of rooms per house, or distance to
employment centers, to predict the 14’th attribute: median dollar
value of the houses.

To run the demo, simply `include("housing.jl")` and run `Housing.main()`.  
The dataset will be automatically downloaded.  You can provide the
initial weights as an optional argument, which should be a pair of
1x13 weight matrix and a scalar bias.  `train` also accepts the
following keyword arguments: `lr` specifies the learning rate,
`epochs` gives number of epochs, and `seed` specifies the random
number seed.  The quadratic loss for the train and test sets will be
printed at every epoch and optimized parameters will be returned.
"""
module Housing
using Knet,AutoGrad,ArgParse

function main(args=ARGS)
    global w, dtrn, dtst
    s = ArgParseSettings()
    s.description="housing.jl (c) Deniz Yuret, 2016. Linear regression model for the Housing dataset from the UCI Machine Learning
Repository"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1)
        ("--epochs"; arg_type=Int; default=20)
        ("--lr"; arg_type=Float64; default=0.1)
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o)
    o[:seed] > 0 && srand(o[:seed])
    w = Any[KnetArray(0.1*randn(1,13)), 0.0]
    (xtrn,ytrn,xtst,ytst) = loaddata()
    println((:epoch,0,:trn,loss(w,xtrn,ytrn),:tst,loss(w,xtst,ytst)))
    @time train(w, xtrn, ytrn; lr=o[:lr], epochs=o[:epochs])
    println((:epoch,o[:epochs],:trn,loss(w,xtrn,ytrn),:tst,loss(w,xtst,ytst)))
end

predict(w,x)=(w[1]*x.+w[2])

loss(w,x,y)=(sum(abs2(y-predict(w,x))) / size(x,2))

lossgradient = grad(loss)

function train(w, x, y; lr=.1, epochs=20)
    for epoch=1:epochs
        g = lossgradient(w, x, y)
        for i in 1:length(w)
            w[i] -= lr * g[i]
        end
    end
    return w
end

function loaddata()
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    file=Pkg.dir("Knet/data/housing.data")
    if !isfile(file)
        info("Downloading $url to $file")
        download(url, file)
    end
    data = readdlm(file)'
    @show size(data) # (14,506)
    x = data[1:13,:]
    y = data[14:14,:]
    x = (x .- mean(x,2)) ./ std(x,2) # Data normalization
    r = randperm(size(x,2))          # trn/tst split
    xtrn=KnetArray(x[:,r[1:400]])
    ytrn=KnetArray(y[:,r[1:400]])
    xtst=KnetArray(x[:,r[401:end]])
    ytst=KnetArray(y[:,r[401:end]])
    (xtrn, ytrn, xtst, ytst)
end

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end # module Housing
