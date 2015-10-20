using Knet, ArgParse

function copyseq(args=ARGS)
    info("Learning to copy sequences to test the S2S model.")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("datafiles"; nargs='+'; required=true; help="First file used for training")
        ("--epochs"; arg_type=Int; default=10)
        ("--hidden"; arg_type=Int; default=100)
        ("--batchsize"; arg_type=Int; default=128)
        ("--gclip"; arg_type=Float64; default=10.0)
        ("--lr"; arg_type=Float64; default=2.0)
        ("--ftype"; default="Float32")
        ("--dense"; action=:store_true)
        ("--getnorm"; action=:store_true)
        ("--getloss"; action=:store_true)
        ("--gcheck"; arg_type=Int; default=0)
        ("--seed"; arg_type=Int; default=42)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
    global data = Any[]
    dict = [ Dict{Any,Int32}() for i=1:2 ]
    for f in datafiles
        push!(data, S2SData(f, f; batch=batchsize, ftype=eval(parse(ftype)), dense=dense, dict1=dict[1], dict2=dict[2])) # , stop=6400)) #DBG
    end
    global model = S2S(lstm; hidden=hidden, vocab=length(dict[2]), winit=Uniform(-.05,.05))
    setopt!(model; lr=lr)

    perp = zeros(length(data))
    maxnorm = (getnorm ? [0.0, 0.0] : nothing)
    losscnt = (getloss ? [0.0, 0.0] : nothing)
    t0 = time_ns()
    println("epoch  secs    ptrain  pvalid  ptest.. ltrain  wnorm  gnorm")
    for epoch=1:epochs
        train(model, data[1], softloss; gclip=gclip, maxnorm=fill!(maxnorm,0), losscnt=fill!(losscnt,0))
        losscnt != nothing && (perp[1] = exp(losscnt[1]/losscnt[2]))
        # @show losscnt
        for d=2:length(data)
            loss = test(model, data[d], softloss)
            perp[d] = exp(loss)
        end
        myprint(epoch, (time_ns()-t0)/1e9, perp..., (maxnorm==nothing ? [] : maxnorm)...)
        gcheck > 0 && gradcheck(model, data[1], softloss; gcheck=gcheck)
    end
    return (perp..., (maxnorm==nothing ? [] : maxnorm)...)
end

myprint(a...)=(for x in a; @printf("%-6g ",x); end; println(); flush(STDOUT))

!isinteractive() && !isdefined(:load_only) && copyseq(ARGS)


### DEAD CODE

    # info("Warm-up epoch")
    # f=datafiles[1]; mini = S2SData(f, f; batch=batchsize, ftype=eval(parse(ftype)), dense=dense, dict1=dict[1], dict2=dict[2], stop=3200) #DBG
    # @date train(model, mini, softloss; gcheck=gcheck, gclip=gclip, getnorm=getnorm, getloss=getloss) #DBG: pretrain to compile for timing
    # info("Starting profile")
