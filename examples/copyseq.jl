using Knet, ArgParse

function copyseq(args=ARGS)
    info("Learning to copy sequences to test the S2S model.")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("datafiles"; nargs='+'; required=true; help="First file used for training")
        ("--batchsize"; arg_type=Int; default=20)
        ("--ftype"; default="Float32")
        ("--dense"; action=:store_true)
        ("--epochs"; arg_type=Int; default=10)
        ("--hidden"; arg_type=Int; default=100)
        ("--gcheck"; arg_type=Int; default=0)
        ("--gclip"; arg_type=Float64; default=10.0)
        ("--lr"; arg_type=Float64; default=2.0)
        ("--seed"; arg_type=Int; default=42)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
    data = Any[]
    dict = [ Dict{Any,Int32}() for i=1:2 ]
    for f in datafiles
        push!(data, S2SData(f, f; batch=batchsize, ftype=eval(parse(ftype)), dense=dense, dict1=dict[1], dict2=dict[2]))
    end
    # length(data)==1 && push!(data, data[1]) # If no test data specified use the training data
    global model = S2S(lstm; hidden=hidden, vocab=length(dict[2]))
    setopt!(model; lr=lr)
    wmax = gmax = 0
    perp = zeros(length(data))
    train(model, data[1], softloss; gcheck=gcheck, gclip=gclip) #DBG: pretrain to compile for timing
    
    @time for epoch=1:epochs
        (loss,wmax,gmax) = train(model, data[1], softloss; gcheck=gcheck, gclip=gclip)
        perp[1] = exp(loss)
        for d=2:length(data)
            loss = test(model, data[d], softloss)
            perp[d] = exp(loss)
        end
        println((epoch,perp...,loss,wmax,gmax))
    end
    return (perp..., wmax, gmax)
end

!isinteractive() && !isdefined(:load_only) && copyseq(ARGS)

