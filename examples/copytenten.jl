using Knet, ArgParse

function copytenten(args=ARGS)
    info("Learning to copy sequences to test the S2S model.")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--dictfile"; required=true)
        ("--hidden"; arg_type=Int; default=1024)
        ("--batchsize"; arg_type=Int; default=128)
        ("--epochsize"; arg_type=Int; default=typemax(Int))
        ("--lossreport"; arg_type=Int; default=100000)
        ("--maxlen"; arg_type=Int; default=40)
        ("--gclip"; arg_type=Float64; default=5.0)
        ("--lr"; arg_type=Float64; default=2.0)
        ("--ftype"; default="Float32")
        ("--winit"; default="Gaussian(0,0.01)")
        ("--dense"; action=:store_true)
        ("--fast"; help="skip norm and loss calculations."; action=:store_true)
        ("--seed"; arg_type=Int; default=42)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
    global data = S2SData(`tenten.pl -l $maxlen`; batchsize=batchsize, epochsize=epochsize, ftype=eval(parse(ftype)), dense=dense, dict=dictfile)
    vocab = maxtoken(data,2)
    global model = S2S(lstm; hidden=hidden, vocab=vocab, winit=eval(parse(winit)))
    setopt!(model; lr=lr)
    (maxnorm,losscnt) = fast ? (nothing,zeros(2)) : (zeros(2),zeros(2))
    train(model, data, softloss; gclip=gclip, maxnorm=maxnorm, losscnt=losscnt, lossreport=lossreport)
    Knet.s2s_lossreport(losscnt, batchsize, 0)
end

!isinteractive() && !isdefined(:load_only) && copytenten(ARGS)
