using Knet, ArgParse

function copyseq(args=ARGS)
    info("Learning to copy sequences to test the S2S model.")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("datafiles"; nargs='+'; required=true; help="First file used for training")
        ("--dictfile"; help="Dictionary file, first datafile used if not specified")
        ("--epochs"; arg_type=Int; default=1)
        ("--hidden"; arg_type=Int; default=100)
        ("--batchsize"; arg_type=Int; default=128)
        ("--lossreport"; arg_type=Int; default=0)
        ("--gclip"; arg_type=Float64; default=10.0)
        ("--lr"; arg_type=Float64; default=2.0)
        ("--ftype"; default="Float32")
        ("--winit"; default="Gaussian(0,0.01)")
        ("--dense"; action=:store_true)
        ("--fast"; help="skip norm and loss calculations."; action=:store_true)
        ("--gcheck"; arg_type=Int; default=0)
        ("--seed"; arg_type=Int; default=42)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
    dict = (dictfile == nothing ? datafiles[1] : dictfile)
    global data = Any[]
    for f in datafiles
        push!(data, S2SData(f; batchsize=batchsize, ftype=eval(parse(ftype)), dense=dense, dict=dict))
    end
    vocab = maxtoken(data[1],2)
    global model = S2S(lstm; hidden=hidden, vocab=vocab, winit=eval(parse(winit)))
    setopt!(model; lr=lr)

    perp = zeros(length(data))
    (maxnorm,losscnt) = fast ? (nothing,nothing) : (zeros(2),zeros(2))
    t0 = time_ns()
    println("epoch  secs    ptrain  ptest.. wnorm  gnorm")
    for epoch=1:epochs
        fast || (fill!(maxnorm,0); fill!(losscnt,0))
        train(model, data[1], softloss; gclip=gclip, maxnorm=maxnorm, losscnt=losscnt, lossreport=lossreport)
        fast || (perp[1] = exp(losscnt[1]/losscnt[2]))
        for d=2:length(data)
            loss = test(model, data[d], softloss)
            perp[d] = exp(loss)
        end
        myprint(epoch, (time_ns()-t0)/1e9, perp..., (fast ? [] : maxnorm)...)
        gcheck > 0 && gradcheck(model, data[1], softloss; gcheck=gcheck)
    end
    return (fast ? (perp...) :  (perp..., maxnorm...))
end

myprint(a...)=(for x in a; @printf("%-6g ",x); end; println(); flush(STDOUT))

!isinteractive() && !isdefined(:load_only) && copyseq(ARGS)


### DEAD CODE

    # info("Warm-up epoch")
    # f=datafiles[1]; mini = S2SData(f, f; batch=batchsize, ftype=eval(parse(ftype)), dense=dense, dict1=dict[1], dict2=dict[2], stop=3200)
    # @date train(model, mini, softloss; gcheck=gcheck, gclip=gclip, getnorm=getnorm, getloss=getloss) # pretrain to compile for timing
    # info("Starting profile")
