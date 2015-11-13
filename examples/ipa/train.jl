using Knet, ArgParse, JLD
using Knet: clear!

@knet function droplstm(x0; fbias=1, drop1=0.2, drop2=0.2, o...)
    x = drop(x0; pdrop=drop1)
    input  = add2(x,h; o..., f=sigm)
    forget = add2(x,h; o..., f=sigm, binit=Constant(fbias))
    output = add2(x,h; o..., f=sigm)
    newmem = add2(x,h; o..., f=tanh)
    ig = mul(input,newmem)
    fc = mul(forget,cell)
    cell = add(ig,fc)
    tc = tanh(cell)
    h  = mul(tc,output)
    h0 = drop(h; pdrop=drop2)
end

function ipa(args=ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; required=true; help="First pair used for training, second pair if any for dev, others test")
        ("--dictfiles"; nargs='+'; required=true; help="Dictionary file pair")
        ("--loadfile"; help="model input file name")
        ("--bestfile"; help="model best file name")
        ("--savefile"; help="model output file name")
        ("--epochs"; arg_type=Int; default=30; help="number of epochs to train")
        ("--batchsize"; arg_type=Int; default=128; help="training minibatch size")
        ("--lossreport"; arg_type=Int; default=100000; help="frequency of loss output")
        ("--hidden"; arg_type=Int; default=512; help="hidden layer size")
        ("--lr"; arg_type=Float64; default=2.0; help="learning rate")
        ("--decay"; arg_type=Float64; default=1.0; help="lr *= decay if an epoch does not improve dev")
        ("--gclip"; arg_type=Float64; default=5.0; help="gradient clipping")
        ("--drop1"; arg_type=Float64; default=0.8; help="dropout probability for input embedding")
        ("--drop2"; arg_type=Float64; default=0.4; help="dropout probability for output embedding")
        ("--fbias"; arg_type=Float64; default=1.0; help="forget gate bias for lstm")
        ("--ftype"; default="Float32"; help="floating point type")
        # ("--winit"; arg_type=Float64; default=0.15)
        ("--usesparse"; action=:store_true; help="use sparse matrices if true")
        ("--fast"; help="skip norm and loss calculations."; action=:store_true)
        ("--gcheck"; arg_type=Int; default=0; help="perform gradient checking on gcheck samples if nonzero")
        ("--seed"; arg_type=Int; default=42; help="random seed")
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
    @assert length(dictfiles)==2
    global data = Any[]
    for i in 1:2:length(datafiles)
        push!(data, S2SData(datafiles[i], datafiles[i+1]; dict1=dictfiles[1], dict2=dictfiles[2], dense=!usesparse))
    end
    vocab = maxtoken(data[1],2)
    global model = loadfile == nothing ?
    S2S(droplstm; fbias=fbias, hidden=hidden, vocab=vocab, winit=Xavier(), drop1=drop1, drop2=drop2) :
    gpucopy(load(loadfile, "model"))
    lrate = lr; setopt!(model; lr=lrate)

    perp = zeros(length(data))
    (maxnorm,losscnt) = fast ? (nothing,nothing) : (zeros(2),zeros(2))
    lastloss = bestloss = Inf
    t0 = time_ns()
    for epoch=1:epochs
        fast || (fill!(maxnorm,0); fill!(losscnt,0))
        train(model, data[1], softloss; gclip=gclip, maxnorm=maxnorm, losscnt=losscnt, lossreport=lossreport) # use first pair for training
        fast || (perp[1] = exp(losscnt[1]/losscnt[2]))
        for d=2:length(data)
            loss = test(model, data[d], softloss)
            perp[d] = exp(loss)
            if d==2              # use second pair for validation
                bestfile!=nothing && loss < bestloss && (bestloss=loss; save(bestfile, "model", clear!(cpucopy(model)); compress=true))
                loss < lastloss || (@show lrate = decay*lrate; setopt!(model; lr=lrate))
                lastloss = loss
            end
        end
        println("epoch  secs    ptrain  ptest.. wnorm  gnorm")
        myprint(epoch, (time_ns()-t0)/1e9, perp..., (fast ? [] : maxnorm)...)
        gcheck > 0 && gradcheck(model, data[1], softloss; gcheck=gcheck)
    end
    savefile==nothing || save(savefile, "model", clear!(cpucopy(model)); compress=true)
    return (fast ? (perp...) :  (perp..., maxnorm...))
end

myprint(a...)=(for x in a; @printf("%-6g ",x); end; println(); flush(STDOUT))

!isinteractive() && !isdefined(:load_only) && ipa(ARGS)
