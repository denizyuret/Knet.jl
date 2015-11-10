using Knet, ArgParse
import Base: start, next, done

function ncelm(args=ARGS)
    info("NCE language model")
    isa(args, AbstractString) && (args=split(args))
    opts = nce_parse_commandline(args)
    println(opts)
    opts["seed"] > 0 && setseed(opts["seed"])
    opts["dropout"] > 0 && error("TODO: implement dropout")
    ftype = opts["float64"]?Float64:Float32
    dict = Dict{Any,Int32}()
    global data = Any[]
    for f in opts["datafiles"]
        push!(data, LMData(f;  dict=dict, 
                           batch=opts["batch_size"], 
                           seqlen=opts["seq_length"], 
                           ftype=ftype,
                           dense=opts["dense"]))
    end

    vocab_size = length(dict)
    psample = fill(ftype(1/vocab_size), vocab_size) # TODO: compute unigram instead

    global net = NCE(nce_rnn; layers = opts["layers"], rnn_size = opts["rnn_size"], vocab = vocab_size, psample=psample, nsample=opts["nsample"])
    lr = opts["lr"]
    setopt!(net, lr=lr, init = Uniform(-opts["init_weight"], opts["init_weight"]))

    perp = zeros(length(data))
    l=zeros(2); m=zeros(2)
    for ep=1:opts["max_max_epoch"]
        ep > opts["max_epoch"] && (lr /= opts["decay"]; setopt!(net, lr=lr))
        train(net, data[1], nothing; psample=psample, nsample=opts["nsample"], gclip=opts["max_grad_norm"], keepstate=true, losscnt=fill!(l,0), maxnorm=fill!(m,0))
        opts["gcheck"]>0 && gradcheck(net,data[1],softloss; gcheck=opts["gcheck"])
        perp[1] = exp(l[1]/l[2])
        for idata = 2:length(data)
            ldev = test(net, data[idata], softloss; keepstate=true)
            perp[idata] = exp(ldev)
        end
        @show (ep, perp..., m..., lr)
    end
    return (perp..., m...)
end

@knet function nce_rnn(word; layers=1, rnn_size=0, rnn_type=lstm, o...)
    wvec = wdot(word; o..., out=rnn_size)
    yrnn = repeat(wvec; o..., frepeat=rnn_type, nrepeat=layers, out=rnn_size)
end

type LMData; data; dict; batchsize; seqlength; ftype; dense; x; y; end

function LMData(fname::AbstractString; batch=20, seqlen=20, dict=Dict{Any,Int32}(), ftype=Float32, dense=false)
    data = Int32[]
    open(fname) do f 
        for l in eachline(f)
            for w in split(l)
                push!(data, get!(dict, w, 1+length(dict)))
            end
            push!(data, get!(dict, "<eos>", 1+length(dict))) # end-of-sentence
        end
    end
    info("Read $fname: $(length(data)) words, $(length(dict)) vocab.")
    LMData(data, dict, batch, seqlen, ftype, dense, nothing, nothing)
end

function next(d::LMData,state)                              	# d.data is the whole corpus represented as a sequence of Int32's
    (nword, eos) = state                                        # eos is true if last output was nothing
    if !eos && (nword % (d.batchsize * d.seqlength) == 0)
        return (nothing, (nword, true))                         # output nothing indicating end of sequence
    end
    segsize = div(length(d.data), d.batchsize)                  # we split it into d.batchsize roughly equal sized segments
    offset = div(nword, d.batchsize)                            # this is how many words have been served so far from each segment
    if issparse(d.x)
        for b = 1:d.batchsize
            idata = (b-1)*segsize + offset
            d.x.rowval[b] = d.data[idata+1]
            d.y.rowval[b] = d.data[idata+2]
        end
    else
        fill!(d.x, 0)
        fill!(d.y, 0)
        for b = 1:d.batchsize
            idata = (b-1)*segsize + offset
            d.x[d.data[idata+1], b] = 1
            d.y[d.data[idata+2], b] = 1
        end
    end
    ((d.x, d.y), (nword + d.batchsize, false))	# each call to next will deliver a single word from each of the d.batchsize segments
end

# The state indicates number of words served, and whether the last output was end-of-sequence nothing
function start(d::LMData)
    ndict = length(d.dict)
    if d.x == nothing || size(d.x) != (ndict, d.batchsize)
        if d.dense
            d.x = zeros(d.ftype, ndict, d.batchsize)
            d.y = zeros(d.ftype, ndict, d.batchsize)
        else
            d.x = sponehot(d.ftype, ndict, d.batchsize)
            d.y = sponehot(d.ftype, ndict, d.batchsize)
        end
    end
    return (0, true)
end

function done(d::LMData,state)
    # nword is the number of sequences served
    # stop if there is not enough data for another full sequence
    (nword, eos) = state
    eos && nword + d.batchsize * d.seqlength > length(d.data)
end

function nce_parse_commandline(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--batch_size"
        help = "minibatch size"
        arg_type = Int
        default = 128
        "--seq_length"
        help = "length of the input sequence"
        arg_type = Int
        default = 20
        "--layers"
        help = "number of lstm layers"
        arg_type = Int
        default = 1
        "--decay"
        help = "divide learning rate by this every epoch after max_epoch"
        arg_type = Float64
        default = 2.0
        "--rnn_size"
        help = "size of the lstm"
        arg_type = Int
        default = 200
        "--nsample"
        help = "number of noise samples per minibatch"
        arg_type = Int
        default = 100
        "--dropout"
        help = "dropout units with this probability"
        arg_type = Float64
        default =  0.0
        "--init_weight"
        help = "initialize weights uniformly in [-init_weight,init_weight]"
        arg_type = Float64
        default =  0.1
        "--lr"
        help = "learning rate"
        arg_type = Float64
        default = 1.0
        "--vocab_size"
        help = "size of the vocabulary"
        arg_type = Int
        default = 10000
        "--max_epoch"
        help = "number of epochs at initial lr"
        arg_type = Int
        default = 10000 # 4
        "--max_max_epoch"
        help = "number of epochs to train"
        arg_type = Int
        default = 1 # 13
        "--max_grad_norm"
        help = "gradient clip"
        arg_type = Float64
        default = 5.0
        "--gcheck"
        help = "gradient check this many units per layer"
        arg_type = Int
        default = 0
        "--seed"
        help = "random seed, use 0 to turn off"
        arg_type = Int
        default = 42
        "--float64"
        help = "Use Float64 (Float32 default)"
        action = :store_true
        "--dense"
        help = "Use dense matrix ops (sparse ops default)"
        action = :store_true
        "datafiles"
        help = "corpus files: first one will be used for training"
        nargs = '+'
        required = true
    end
    opts = parse_args(args,s)
    return opts
end

!isinteractive() && !isdefined(:load_only) && ncelm(ARGS)


# version 099b284
# [dy_052@hpc3004 examples]$ julia ncelm.jl ptb.valid.txt ptb.test.txt 
# WARNING: requiring "Knet" in module "Main" did not define a corresponding module.
# INFO: NCE language model
# Dict{AbstractString,Any}("rnn_size"=>200,"lr"=>1.0,"decay"=>2.0,"dense"=>false,"batch_size"=>20,"float64"=>false,"dropout"=>0.0,"max_grad_norm"=>5.0,"max_epoch"=>10000,"init_weight"=>0.1,"gcheck"=>0,"layers"=>1,"vocab_size"=>10000,"seq_length"=>20,"nsample"=>100,"seed"=>42,"max_max_epoch"=>1,"datafiles"=>Any["ptb.valid.txt","ptb.test.txt"])
# INFO: Read ptb.valid.txt: 73760 words, 6022 vocab.
# INFO: Read ptb.test.txt: 82430 words, 7596 vocab.
#  19.979308 seconds (27.21 M allocations: 1.133 GB, 3.39% gc time)
#   6.901726 seconds (8.21 M allocations: 342.865 MB, 2.84% gc time)
# (ep,perp...,m...,lr) = (1,1.0383780365714608,816.3109741210938,999.4002685546875,38.147953033447266,1.0)
