"""
charlm.jl: Knet8 version (c) Emre Yolcu, Deniz Yuret, 2016

This example implements an LSTM network for training and testing
character-level language models inspired by ["The Unreasonable
Effectiveness of Recurrent Neural
Networks"](http://karpathy.github.io/2015/05/21/rnn-effectiveness) from
the Andrej Karpathy blog.  The model can be trained with different
genres of text, and can be used to generate original text in the same
style.

Example usage:

* `julia charlm.jl`: trains a model using 'The Complete Works of
  Shakespeare' using default options.

* `julia charlm.jl --gpu`: uses the GPU for training.

* `julia charlm.jl --data foo.txt`: uses foo.txt to train instead.

* `julia charlm.jl --data foo.txt bar.txt`: uses foo.txt for training
  and bar.txt for validation.  Any number of files can be specified,
  the first two will be used for training and validation, the rest for
  testing.

* `julia charlm.jl --best foo.jld --save bar.jld`: saves the best
  model (according to validation set) to foo.jld, last model to
  bar.jld.

* `julia charlm.jl --load foo.jld --generate 1000`: generates 1000
  characters from the model in foo.jld.

* `julia charlm.jl --help`: describes all available options.
    
"""
module CharLM

using Knet,AutoGrad,ArgParse,Compat

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="charlm.jl (c) Emre Yolcu, Deniz Yuret, 2016. Character level language model based on http://karpathy.github.io/2015/05/21/rnn-effectiveness."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
        ("--loadfile"; help="Initialize model from file")
        ("--savefile"; help="Save final model to file")
        ("--bestfile"; help="Save best model to file")
        ("--generate"; arg_type=Int; default=0; help="If non-zero generate given number of characters.")
        ("--hidden"; arg_type=Int; default=256; help="Size of the LSTM internal state.")
        ("--embedding"; arg_type=Int; default=256; help="Size of the embedding vector.")
        ("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=128; help="Number of sequences to train on in parallel.")
        ("--seqlength"; arg_type=Int; default=100; help="Number of steps to unroll the network for.")
        ("--decay"; arg_type=Float64; default=0.9; help="Learning rate decay.")
        ("--lr"; arg_type=Float64; default=1.0; help="Initial learning rate.")
        ("--gclip"; arg_type=Float64; default=5.0; help="Value to clip the gradient norm at.")
        ("--keepstate"; action=:store_true; help="Keep state between iterations.")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--seed"; arg_type=Int; default=42; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        #TODO ("--dropout"; arg_type=Float64; default=0.0; help="Dropout probability.")
        #TODO ("--nlayer"; arg_type=Int; default=1; help="Number of LSTM layers.")
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    if any(f->(o[f]!=nothing), (:loadfile, :savefile, :bestfile))
        isdir(Pkg.dir("JLD")) || error("Please Pkg.add(\"JLD\") to load or save files.")
        eval(Expr(:using,:JLD))
    end

    # we initialize a model from loadfile, train using datafiles (both optional).
    # if the user specifies neither, train a model using shakespeare.
    isempty(o[:datafiles]) && o[:loadfile]==nothing && push!(o[:datafiles],shakespeare())

    # read text and report lengths
    text = map((@compat readstring), o[:datafiles])
    !isempty(text) && info("Chars read: $(map((f,c)->(basename(f),length(c)),o[:datafiles],text))")

    # vocab (char_to_index) comes from the initial model if there is one, otherwise from the datafiles.
    # if there is an initial model make sure the data has no new vocab
    if o[:loadfile]==nothing
        vocab = Dict{Char,Int}()
        for t in text, c in t; get!(vocab, c, 1+length(vocab)); end
        model = weights(length(vocab), o[:hidden], o[:embedding])
    else
        info("Loading model from $(o[:loadfile])")
        vocab = load(o[:loadfile], "vocab") 
        for t in text, c in t; haskey(vocab, c) || error("Unknown char $c"); end
        model = load(o[:loadfile], "model")
    end
    info("$(length(vocab)) unique chars.")
    if !isempty(text)
        train!(model, text, vocab, o)
    end
    if o[:savefile] != nothing
        info("Saving last model to $(o[:savefile])")
        save(o[:savefile], "model", model, "vocab", vocab)
    end
    if o[:generate] > 0
        generate(model, vocab, o[:generate])
    end
end


function train!(model, text, vocab, o)
    atype = eval(parse(o[:atype]))
    for (k,v) in model; model[k] = convert(atype,v); end
    h0 = c0 = convert(atype, zeros(o[:hidden], o[:batchsize])); s0=Any[h0,c0]
    data = map(t->minibatch(t, vocab, o[:batchsize]), text)
    @time losses = map(d->loss(model,d,s0), data)
    println((:epoch,0,:loss,losses...))
    devset = ifelse(length(data) > 1, 2, 1)
    devlast = devbest = losses[devset]
    lr = o[:lr]
    for epoch=1:o[:epochs]
        @time train1(model, data[1], s0; slen=o[:seqlength], lr=lr, gclip=o[:gclip], keepstate=o[:keepstate])
        o[:fast] && continue
        @time losses = map(d->loss(model,d,s0), data)
        println((:epoch,epoch,:loss,losses...))
        if o[:gcheck] > 0
            gradcheck(loss, model, data[1], s0; gcheck=o[:gcheck], range=1:o[:seqlength])
        end
        devloss = losses[devset]
        if devloss < devbest
            devbest = devloss
            if o[:bestfile] != nothing
                info("Saving best model to $(o[:bestfile])")
                save(o[:bestfile], "model", model, "vocab", vocab)
            end
        end
        if devloss > devlast
            lr *= o[:decay]
            info("New learning rate: $lr")
        end
        devlast = devloss
    end
    if o[:fast]
        @time losses = map(d->loss(model,d,s0), data)
        println((:epoch,o[:epochs],:loss,losses...))
    end
end    

function train1(w, x, state; slen=100, lr=1.0, gclip=0.0, keepstate=false)
    keepstate && (state = copy(state))
    for t = 1:slen:length(x)-slen
        r = t:t+slen-1
        g = lossgradient(w, x, state; range=r, keepstate=keepstate)
        gscale = lr
        if gclip > 0
            gnorm = sqrt(mapreduce(a->vecnorm(a)^2, +, 0, values(g)))
            if gnorm > gclip
                gscale *= gclip / gnorm
            end
        end
        for k in keys(g)
            w[k] -= gscale * g[k]
            # TODO: try axpy! to see if it is worth it
        end
    end
end

# Given parameters w, sequence x, and hidden-cell pair state, returns loss.
function loss(w, x, state; range=1:length(x)-1, keepstate=false)
    (h,c) = state
    loss = 0.0; xcnt = 0
    atype = typeof(getval(w[:W_embedding]))
    xcurr = convert(atype, x[first(range)])
    for t in range
        xt = w[:W_embedding] * xcurr
        (h,c) = lstm(w, xt, h, c)
        ypred = w[:W_predict] * h .+ w[:b_predict]
        ynorm = logp(ypred)
        xnext = convert(atype, x[t+1])
        loss += sum(xnext .* ynorm)
        xcnt += size(ynorm,2)
        xcurr = xnext
    end
    if keepstate
        state[1]=getval(h); state[2]=getval(c)
    end
    return -loss/xcnt
end

lossgradient = grad(loss)

# TODO: implement vcat for KnetArray and try various concat versions for efficiency.
function lstm(w, input, hidden, cell)
    ingate  = sigm(w[:Wx_ingate]  * input .+ w[:Wh_ingate] * hidden .+ w[:b_ingate]) # in fact we can probably combine these four operations into one
    forget  = sigm(w[:Wx_forget]  * input .+ w[:Wh_forget] * hidden .+ w[:b_forget]) # then use indexing, or (better) subarrays to get individual gates
    outgate = sigm(w[:Wx_outgate] * input .+ w[:Wh_outgate] * hidden .+ w[:b_outgate])
    change  = tanh(w[:Wx_change]  * input .+ w[:Wh_change] * hidden .+ w[:b_change])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return hidden, cell
end

initstate(h,b)=Any[zeros(h,b), zeros(h,b)]

function weights(vocabsize,hiddensize,embedsize)
    w = Dict()
    for gate in (:ingate, :forget, :outgate, :change)
        w[Symbol("Wx_$gate")] = xavier(hiddensize, embedsize)
        w[Symbol("Wh_$gate")] = xavier(hiddensize, hiddensize)
        w[Symbol("b_$gate")] = (gate == :forget ? ones : zeros)(hiddensize, 1)
    end
    w[:W_embedding] = xavier(embedsize, vocabsize)
    w[:W_predict]   = xavier(vocabsize, hiddensize)
    w[:b_predict]   = zeros(vocabsize, 1)
    return w
end

function xavier(a...)
    w = rand(a...)
     # The old implementation was not right for fully connected layers:
     # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
        fanin = div(length(w), fanout)
    end
    # See: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
end

function generate(model, vocab, nchar)
    index_to_char = Array(Char, length(vocab))
    for (k,v) in vocab; index_to_char[v] = k; end
    w = Dict()
    for (k,v) in model; w[k] = Array(v); end
    h = c = zeros(w[:b_ingate])
    xcurr = zeros(w[:b_predict])
    index = 1
    for t = 1:nchar
        xt = w[:W_embedding] * xcurr
        (h,c) = lstm(w, xt, h, c)
        ypred = w[:W_predict] * h .+ w[:b_predict]
        xcurr[index,1] = 0
        index = sample(exp(logp(ypred)))
        xcurr[index,1] = 1
        print(index_to_char[index])
    end
    println()
end

function sample(p)
    r = rand(Float32)
    for c = 1:length(p)
        r -= p[c]
        r < 0 && return c
    end
end

function shakespeare()
    file = Pkg.dir("Knet/data/100.txt")
    if !isfile(file)
        info("Downloading 'The Complete Works of William Shakespeare'")
        url = "http://www.gutenberg.org/files/100/100.txt"
        download(url,file)
    end
    return file
end

function minibatch(chars, char_to_index, batch_size)
    nbatch = div(length(chars), batch_size)
    vocab_size = length(char_to_index)
    data = Any[]
    for i = 1:nbatch
        d = falses(vocab_size, batch_size) # use BitArray until we implement sparse
        for j = 1:batch_size
            d[char_to_index[chars[i + nbatch * (j - 1)]], j] = 1
        end
        push!(data, d)  # do not convert here, we don't need all data on gpu! convert(o[:atype], d))
    end
    return data
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia charlm.jl --epochs 10
# julia> CharLM.main("--epochs 10")
!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end  # module

# SAMPLE RUN 65f57ff+ Wed Sep 14 10:02:30 EEST 2016
#
# charlm.jl (c) Emre Yolcu, Deniz Yuret, 2016. Character level language model based on http://karpathy.github.io/2015/05/21/rnn-effectiveness.
# opts=(:keepstate,false)(:lr,1.0)(:atype,"KnetArray{Float32}")(:savefile,nothing)(:loadfile,nothing)(:generate,0)(:bestfile,nothing)(:embedding,256)(:gclip,5.0)(:hidden,256)(:epochs,3)(:decay,0.9)(:gcheck,0)(:seqlength,100)(:seed,42)(:batchsize,128)(:datafiles,Any["10.txt"])(:fast,true)
# INFO: Chars read: [("10.txt",425808)]
# INFO: 87 unique chars.
#   2.156358 seconds (2.31 M allocations: 237.913 MB, 2.30% gc time)
# (:epoch,0,:loss,4.465127425659868)
#   6.287736 seconds (9.54 M allocations: 574.703 MB, 2.84% gc time)
#   6.272144 seconds (9.54 M allocations: 574.633 MB, 2.80% gc time)
#   6.277462 seconds (9.54 M allocations: 574.637 MB, 2.86% gc time)
#   2.165516 seconds (2.34 M allocations: 238.323 MB, 2.56% gc time)
# (:epoch,3,:loss,3.226540256084356)


### SAMPLE OUTPUT (with head -10000 100.txt):
# julia> CharLM.main("--gpu --data 10.txt")
# opts=(:lr,1.0)(:savefile,nothing)(:loadfile,nothing)(:dropout,0.0)(:generate,0)(:bestfile,nothing)(:embedding,256)(:gclip,5.0)(:hidden,256)(:epochs,10)(:nlayer,1)(:decay,0.9)(:gpu,true)(:seqlength,100)(:seed,42)(:batchsize,128)(:datafiles,Any["10.txt"])
# INFO: Chars read: [("10.txt",425808)]
# INFO: 87 unique chars.
# (0,4.465127425659868)
#   2.182693 seconds (2.36 M allocations: 240.394 MB, 1.74% gc time)
#   7.861079 seconds (10.12 M allocations: 601.311 MB, 1.84% gc time)
# (1,3.3244698245543285)
#   2.159062 seconds (2.35 M allocations: 239.217 MB, 1.80% gc time)
#   6.200085 seconds (9.55 M allocations: 575.573 MB, 2.28% gc time)
# (2,3.24593908969621)
#   2.352389 seconds (2.34 M allocations: 239.381 MB, 1.51% gc time)
#   6.211946 seconds (9.55 M allocations: 575.568 MB, 2.21% gc time)
# (3,3.226540256084356)
