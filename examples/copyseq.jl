using Knet, ArgParse

@knet function copymodel(word; hidden=0, vocab=0)
    wvec = wdot(word; out=hidden)
    hvec = lstm(wvec; out=hidden)
    tvec = wbf(hvec; out=vocab, f=soft) # TODO: find a way to turn this off for encoder
end

function copyseq(args=ARGS)
    info("Learning to copy sequences to test the S2S model.")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("datafile"; help="Input file")
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
    data = S2SData(datafile, datafile; batch=batchsize, ftype=eval(parse(ftype)), dense=dense)
    model = S2S(copymodel; hidden=hidden, vocab=length(data.dict2))
    setopt!(model; lr=lr)
    perp = wmax = gmax = 0
    for epoch=1:epochs
        (loss,wmax,gmax) = train(model, data, softloss; gcheck=gcheck, gclip=gclip)
        perp = exp(loss)
        println((epoch,perp,loss,wmax,gmax))
    end
    return (perp, wmax, gmax)
end

!isinteractive() && !isdefined(:load_only) && copyseq(ARGS)

