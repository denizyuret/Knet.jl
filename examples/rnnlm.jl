# This is to replicate the language modeling experiments from Zaremba,
# Wojciech, Ilya Sutskever, and Oriol Vinyals. "Recurrent neural
# network regularization." arXiv preprint arXiv:1409.2329 (2014).
#
# Usage: julia rnnlm.jl ptb.train.txt ptb.valid.txt ptb.test.txt
# Type julia rnnlm.jl --help for more options

using Knet
module RNNLM
using Knet, ArgParse
using Knet: regs, getp, setp, stack_length, stack_empty!, params

function main(args=ARGS)
    info("RNN language model example from Zaremba et al. 2014.")
    isa(args, AbstractString) && (args=split(args))
    opts = parse_commandline(args)
    println(opts)
    opts["seed"] > 0 && setseed(opts["seed"])
    opts["dropout"] > 0 && error("TODO: implement dropout")
    dict = Dict{Any,Int32}()
    global data = Any[]
    for f in opts["datafiles"]
        push!(data, LMData(f;  dict=dict, 
                           batch=opts["batch_size"], 
                           seqlen=opts["seq_length"], 
                           ftype=opts["float64"]?Float64:Float32, 
                           dense=opts["dense"]))
    end

    vocab_size = length(dict)
    global net = compile(:rnnlm;  layers = opts["layers"], rnn_size = opts["rnn_size"], vocab_size = vocab_size)
    lr = opts["lr"]
    setp(net; lr=lr) 
    init = Uniform(-opts["init_weight"], opts["init_weight"])
    for r in params(net); r.op.init=init; end
    if opts["nosharing"]
        setp(net, :forwoverwrite, false)
        setp(net, :backoverwrite, false)
    end
    perp = zeros(length(data)); l=zeros(2); m=zeros(2)

    for ep=1:opts["max_max_epoch"]
        ep > opts["max_epoch"] && (lr /= opts["decay"]; setp(net; lr=lr))
        train(net, data[1], softloss; gclip=opts["max_grad_norm"], losscnt=fill!(l,0), maxnorm=fill!(m,0))
        perp[1] = exp(l[1]/l[2])
        opts["gcheck"]>0 && gradcheck(net,
                                      f->train(f,data[1],softloss;gcheck=true),
                                      f->test(f,data[1],softloss;gcheck=true);
                                      gcheck=opts["gcheck"])
        for idata = 2:length(data)
            ldev = test(net, data[idata], softloss)
            perp[idata] = exp(ldev)
        end
        @show (ep, perp..., m..., lr)
    end
    return (perp..., m...)
end

@knet function rnnlm(word; layers=0, rnn_size=0, vocab_size=0, rnn_type=:lstm1, o...)
    wvec = wdot(word; o..., out=rnn_size) # 1-3
    yrnn = repeat(wvec; o..., frepeat=rnn_type, nrepeat=layers, out=rnn_size) # 4-40 with 41 copy for return
    return wbf(yrnn; o..., out=vocab_size, f=:soft) # 42-46
end

@knet function lstm1(x; o...)   # need version without fbias
    input  = wbf2(x,h; o..., f=:sigm)
    forget = wbf2(x,h; o..., f=:sigm)
    output = wbf2(x,h; o..., f=:sigm)
    newmem = wbf2(x,h; o..., f=:tanh)
    cell = input .* newmem + cell .* forget
    h  = tanh(cell) .* output
    return h
end

reset_trn!(f;o...)=reset!(f, keepstate=true)

function reset_trn_old!(f; o...)
    if stack_length(f) != length(f)
        info("Stack length: $(stack_length(f)) Regs: $(length(f))")
    end
    stack_empty!(f)
    reset_tst!(f; o...)
    for p in regs(f)            # This is necessary so when back looks at time < 0 the stack gives something back.
        p.dif = nothing
        getp(p,:incr) && fill!(p.dif0, 0)
        setp(p, :forw, true)    # I am not sure why this is necessary
        push!(f,p)
    end
end

function reset_tst!(f; keepstate=false)
    for p in regs(f)
        p.out = keepstate && isdefined(p,:out0) ? p.out0 : nothing
    end
end

# _update_dbg = 0

function train(f, data, loss; gcheck=false, gclip=0, maxnorm=nothing, losscnt=nothing)
    reset_trn!(f)
    ystack = Any[]
    for item in data
        if item != nothing
            (x,ygold) = item
            ypred = sforw(f, x)
            # Knet.netprint(f); error(:ok)
            losscnt != nothing && (losscnt[1] += loss(ypred, ygold); losscnt[2] += 1)
            push!(ystack, copy(ygold))
        else                    # end of sequence
            while !isempty(ystack)
                ygold = pop!(ystack)
                sback(f, ygold, loss)
            end
            #error(:ok)
            gcheck && break # return losscnt[1] leave the loss calculation to test # the parameter gradients are cumulative over the whole sequence
            g = (gclip > 0 || maxnorm!=nothing ? gnorm(f) : 0)
            # global _update_dbg; _update_dbg +=1; _update_dbg > 1 && error(:ok)
            update!(f; gscale=(g > gclip > 0 ? gclip/g : 1))
            if maxnorm != nothing
                w=wnorm(f)
                w > maxnorm[1] && (maxnorm[1]=w)
                g > maxnorm[2] && (maxnorm[2]=g)
            end
            reset_trn!(f; keepstate=true)
        end
    end
    # losscnt[1]/losscnt[2]       # this will give per-token loss, should we do per-sequence instead?
end

function test(f, data, loss; gcheck=false)
    #info("testing")
    sumloss = numloss = 0.0
    reset_tst!(f)
    for item in data
        if item != nothing
            (x,ygold) = item
            ypred = forw(f, x)
            # @show (hash(x),hash(ygold),vecnorm0(ypred))
            sumloss += loss(ypred, ygold)
            numloss += 1
        else
            gcheck && return sumloss
            reset_tst!(f; keepstate=true)
        end
    end
    return sumloss/numloss
end


import Base: start, next, done

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

function parse_commandline(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--batch_size"
        help = "minibatch size"
        arg_type = Int
        default = 20
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
        default = 1 # 4
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
        "--nosharing"
        help = "Do not share register arrays (this is for debugging, should not change results)"
        action = :store_true
        "--preset"
        help = "load one of the preset option combinations"
        arg_type = Int
        default = 0
        "datafiles"
        help = "corpus files: first one will be used for training"
        nargs = '+'
        required = true
    end
    opts = parse_args(args,s)
    # Parameters from the Zaremba implementation:
    if opts["preset"] == 1      # Trains 1h and gives test 115 perplexity.
        opts["batch_size"]=20
        opts["seq_length"]=20
        opts["layers"]=2
        opts["decay"]=2
        opts["rnn_size"]=200
        opts["dropout"]=0
        opts["init_weight"]=0.1
        opts["lr"]=1
        opts["vocab_size"]=10000
        opts["max_epoch"]=4
        opts["max_max_epoch"]=13
        opts["max_grad_norm"]=5
    elseif opts["preset"] == 2  # Train 1 day and gives 82 perplexity.
        opts["batch_size"]=20
        opts["seq_length"]=35
        opts["layers"]=2
        opts["decay"]=1.15
        opts["rnn_size"]=1500
        opts["dropout"]=0.65
        opts["init_weight"]=0.04
        opts["lr"]=1
        opts["vocab_size"]=10000
        opts["max_epoch"]=14
        opts["max_max_epoch"]=55
        opts["max_grad_norm"]=10
    end
    return opts
end

!isinteractive() && !isdefined(Main,:load_only) && main(ARGS)

end # module


### SAMPLE RUN: results may vary because of sparse matrix multiplications.
#
# julia rnnlm.jl --preset 1 ptb.valid.txt ptb.test.txt 
# WARNING: requiring "Options" did not define a corresponding module.
# Dict{AbstractString,Any}("rnn_size"=>200,"lr"=>1,"decay"=>2,"batch_size"=>20,"dropout"=>0,"max_grad_norm"=>5,"preset"=>1,"max_epoch"=>4,"init_weight"=>0.1,"gcheck"=>0,"layers"=>2,"vocab_size"=>10000,"seq_length"=>20,"seed"=>42,"max_max_epoch"=>13,"datafiles"=>Any["data/ptb.valid.txt","data/ptb.test.txt"])
# INFO: Read data/ptb.valid.txt: 73760 words, 6022 vocab.
# INFO: Read data/ptb.test.txt: 82430 words, 7596 vocab.
#  37.057684 seconds (29.17 M allocations: 2.530 GB, 1.31% gc time)
#   9.723167 seconds (11.06 M allocations: 496.906 MB, 1.30% gc time)
# (ep,perp...,wmax,gmax,lr) = (1,964.7574812581911,680.8404134165187,359.66156655473776,182.60932042613925,1)
#  32.492913 seconds (24.07 M allocations: 2.306 GB, 1.39% gc time)
#   9.682416 seconds (11.04 M allocations: 496.187 MB, 1.23% gc time)
# (ep,perp...,wmax,gmax,lr) = (2,533.9658674987758,500.7701607049704,376.4021748281246,29.527137764770796,1)
#  32.433211 seconds (24.07 M allocations: 2.306 GB, 1.39% gc time)
#   9.759406 seconds (11.05 M allocations: 496.439 MB, 1.30% gc time)
# (ep,perp...,wmax,gmax,lr) = (3,390.1504382596744,435.37182169855157,390.84524322483344,30.692745370555468,1)

### Compilation results:
# op(46): input,par,dot; 4xadd2(8); mul,mul,add,tanh,mul; par,dot,par,add,soft,softloss
# tosave(11): 1,3,11,19,27,35,38,39,40,45,46
# toincr(18): 15 par + 3 multi
# toback(45): all except 1
# tmp(6/18): 18 toincr, 6 unique sizes
# out(29/46): 15 par + 11 tosave + 3 tmp
# dif(26/45):

# julia> rnnlm("ptb.test.txt")
# (ep,perp...,wmax,gmax,lr) = (1,694.3535880341253,256.35040516045433,159.35646292641206,1.0)

# function rnnlmModel2(;
#                     layers = 0,
#                     rnn_size = 0,
#                     vocab_size = 0,
#                     )
#     prog = quote                # do we need the prefix?  lstm will do a dot?  yes if you want the same embedding going into each gate.
#         i0 = input()
#         x0 = wdot(i0; out=$rnn_size)
#     end
#     s0 = s1 = :x0
#     for n=1:layers
#         s0 = s1
#         s1 = symbol("x$n")
#         op = :($s1 = lstm($s0; out=$rnn_size))
#         push!(prog.args, op)
#     end
#     prog2 = quote
#         ou = wbf($s1; out=$vocab_size, f=soft)
#     end
#     append!(prog.args, prog2.args)
#     return prog
# end

