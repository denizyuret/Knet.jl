for p in ("Knet","AutoGrad","ArgParse","Compat")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

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

* `julia charlm.jl`: trains a model using its own code.

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
        ("--hidden"; nargs='+'; arg_type=Int; default=[256]; help="Sizes of one or more LSTM layers.")
        ("--embed"; arg_type=Int; default=256; help="Size of the embedding vector.")
        ("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=128; help="Number of sequences to train on in parallel.")
        ("--seqlength"; arg_type=Int; default=100; help="Number of steps to unroll the network for.")
        ("--decay"; arg_type=Float64; default=0.9; help="Learning rate decay.")
        ("--lr"; arg_type=Float64; default=4.0; help="Initial learning rate.")
        ("--gclip"; arg_type=Float64; default=3.0; help="Value to clip the gradient norm at.")
        ("--winit"; arg_type=Float64; default=0.3; help="Initial weights set to winit*randn().")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        #TODO ("--dropout"; arg_type=Float64; default=0.0; help="Dropout probability.")
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    o[:atype] = eval(parse(o[:atype]))
    if any(f->(o[f]!=nothing), (:loadfile, :savefile, :bestfile))
        Pkg.installed("JLD")==nothing && Pkg.add("JLD") # error("Please Pkg.add(\"JLD\") to load or save files.")
        eval(Expr(:using,:JLD))
    end

    # we initialize a model from loadfile, train using datafiles (both optional).
    # if the user specifies neither, train a model using the charlm.jl source code.
    isempty(o[:datafiles]) && o[:loadfile]==nothing && push!(o[:datafiles],@__FILE__) # shakespeare()

    # read text and report lengths
    text = map((@compat readstring), o[:datafiles])
    !isempty(text) && info("Chars read: $(map((f,c)->(basename(f),length(c)),o[:datafiles],text))")

    # vocab (char_to_index) comes from the initial model if there is one, otherwise from the datafiles.
    # if there is an initial model make sure the data has no new vocab
    if o[:loadfile]==nothing
        vocab = Dict{Char,Int}()
        for t in text, c in t; get!(vocab, c, 1+length(vocab)); end
        model = initweights(o[:atype], o[:hidden], length(vocab), o[:embed], o[:winit])
    else
        info("Loading model from $(o[:loadfile])")
        vocab = load(o[:loadfile], "vocab") 
        for t in text, c in t; haskey(vocab, c) || error("Unknown char $c"); end
        model = map(p->convert(o[:atype],p), load(o[:loadfile], "model"))
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
        state = initstate(o[:atype],o[:hidden],1)
        generate(model, state, vocab, o[:generate])
    end
end


function train!(model, text, vocab, o)
    s0 = initstate(o[:atype], o[:hidden], o[:batchsize])
    data = map(t->minibatch(t, vocab, o[:batchsize]), text)
    lr = o[:lr]
    if o[:fast]
        @time (for epoch=1:o[:epochs]
               train1(model, copy(s0), data[1]; slen=o[:seqlength], lr=lr, gclip=o[:gclip])
               end; gpu()>=0 && Knet.cudaDeviceSynchronize())
        return
    end
    losses = map(d->loss(model,copy(s0),d), data)
    println((:epoch,0,:loss,losses...))
    devset = ifelse(length(data) > 1, 2, 1)
    devlast = devbest = losses[devset]
    for epoch=1:o[:epochs]
        @time train1(model, copy(s0), data[1]; slen=o[:seqlength], lr=lr, gclip=o[:gclip])
        @time losses = map(d->loss(model,copy(s0),d), data)
        println((:epoch,epoch,:loss,losses...))
        if o[:gcheck] > 0
            gradcheck(loss, model, copy(s0), data[1], 1:o[:seqlength]; gcheck=o[:gcheck])
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
end    


# sequence[t]: input token at time t
# state is modified in place
function train1(param, state, sequence; slen=100, lr=1.0, gclip=0.0)
    for t = 1:slen:length(sequence)-slen
        range = t:t+slen-1
        gloss = lossgradient(param, state, sequence, range)
        gscale = lr
        if gclip > 0
            gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
            if gnorm > gclip
                gscale *= gclip / gnorm
            end
        end
        for k in 1:length(param)
            # param[k] -= gscale * gloss[k]
            axpy!(-gscale, gloss[k], param[k])
        end
        isa(state,Vector{Any}) || error("State should not be Boxed.")
        # The following is needed in case AutoGrad boxes state values during gradient calculation
        for i = 1:length(state)
            state[i] = AutoGrad.getval(state[i])
        end
    end
end

# param[2k-1,2k]: weight and bias for the k'th lstm layer
# param[end-2]: embedding matrix
# param[end-1,end]: weight and bias for final prediction
function initweights(atype, hidden, vocab, embed, winit)
    param = Array(Any, 2*length(hidden)+3)
    input = embed
    for k = 1:length(hidden)
        param[2k-1] = winit*randn(input+hidden[k], 4*hidden[k])
        param[2k]   = zeros(1, 4*hidden[k])
        param[2k][1:hidden[k]] = 1 # forget gate bias
        input = hidden[k]
    end
    param[end-2] = winit*randn(vocab,embed)
    param[end-1] = winit*randn(hidden[end],vocab)
    param[end] = zeros(1,vocab)
    return map(p->convert(atype,p), param)
end

# state[2k-1,2k]: hidden and cell for the k'th lstm layer
function initstate(atype, hidden, batchsize)
    state = Array(Any, 2*length(hidden))
    for k = 1:length(hidden)
        state[2k-1] = zeros(batchsize,hidden[k])
        state[2k] = zeros(batchsize,hidden[k])
    end
    return map(s->convert(atype,s), state)
end

function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

# s[2k-1,2k]: hidden and cell for the k'th lstm layer
# w[2k-1,2k]: weight and bias for k'th lstm layer
# w[end-2]: embedding matrix
# w[end-1,end]: weight and bias for final prediction
# state is modified in place
function predict(w, s, x)
    x = x * w[end-2]
    for i = 1:2:length(s)
        (s[i],s[i+1]) = lstm(w[i],w[i+1],s[i],s[i+1],x)
        x = s[i]
    end
    return x * w[end-1] .+ w[end]
end

# sequence[t]: input token at time t
# state is modified in place
function loss(param,state,sequence,range=1:length(sequence)-1)
    total = 0.0; count = 0
    atype = typeof(AutoGrad.getval(param[1]))
    input = convert(atype,sequence[first(range)])
    for t in range
        ypred = predict(param,state,input)
        ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
        ygold = convert(atype,sequence[t+1])
        total += sum(ygold .* ynorm)
        count += size(ygold,1)
        input = ygold
    end
    return -total / count
end

lossgradient = grad(loss)

function generate(param, state, vocab, nchar)
    index_to_char = Array(Char, length(vocab))
    for (k,v) in vocab; index_to_char[v] = k; end
    input = oftype(param[1], zeros(1,length(vocab)))
    index = 1
    for t in 1:nchar
        ypred = predict(param,state,input)
        input[index] = 0
        index = sample(exp(logp(ypred)))
        print(index_to_char[index])
        input[index] = 1
    end
    println()
end

function sample(p)
    p = convert(Array,p)
    r = rand()
    for c = 1:length(p)
        r -= p[c]
        r < 0 && return c
    end
end

function shakespeare()
    file = Knet.dir("data","100.txt")
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
    data = [ falses(batch_size, vocab_size) for i=1:nbatch ] # using BitArrays
    cidx = 0
    for c in chars            # safest way to iterate over utf-8 text
        idata = 1 + cidx % nbatch
        row = 1 + div(cidx, nbatch)
        row > batch_size && break
        col = char_to_index[c]
        data[idata][row,col] = 1
        cidx += 1
    end
    return data
end

# To be able to load/save KnetArrays:
if Pkg.installed("JLD") != nothing
    import JLD: writeas, readas
    type KnetJLD; a::Array; end
    writeas(c::KnetArray) = KnetJLD(Array(c))
    readas(d::KnetJLD) = KnetArray(d.a)
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia charlm.jl --epochs 10
# julia> CharLM.main("--epochs 10")
!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end  # module

# Note: 10.txt used in the sample runs below was generated using
#   head -10000 100.txt > 10.txt
# where 100.txt is the file downloaded by shakespeare().



### SAMPLE RUN 74a2e6c+ Mon Sep 19 14:03:10 EEST 2016
### Implemented multi-layer.  Removed the keepstate option fixing it to true.
### Note that winit default changed so I specify it below for comparison.
### The slight difference is due to keepstate.

# julia> CharLM.main("--data 10.txt --winit 0.3 --fast")
# charlm.jl (c) Emre Yolcu, Deniz Yuret, 2016. Character level language model based on http://karpathy.github.io/2015/05/21/rnn-effectiveness.
# opts=(:lr,4.0)(:atype,"KnetArray{Float32}")(:winit,0.3)(:savefile,nothing)(:loadfile,nothing)(:generate,0)(:bestfile,nothing)(:gclip,3.0)(:hidden,[256])(:epochs,3)(:decay,0.9)(:gcheck,0)(:seqlength,100)(:seed,42)(:embed,256)(:batchsize,128)(:datafiles,Any["10.txt"])(:fast,true)
# INFO: Chars read: [("10.txt",425808)]
# INFO: 87 unique chars.
#   1.406687 seconds (1.74 M allocations: 196.799 MB, 2.49% gc time)
# (:epoch,0,:loss,6.1075509900258)
#   4.002638 seconds (6.12 M allocations: 374.188 MB, 2.41% gc time)
#   3.990772 seconds (6.11 M allocations: 374.129 MB, 2.44% gc time)
#   4.006249 seconds (6.12 M allocations: 374.240 MB, 2.53% gc time)
#   1.405878 seconds (1.75 M allocations: 197.059 MB, 2.56% gc time)
# (:epoch,3,:loss,1.8713183968407767)



### SAMPLE RUN 4ce58d1+ Fri Sep 16 12:24:00 EEST 2016
### Transposed everything so getindex does not need to copy

# charlm.jl (c) Emre Yolcu, Deniz Yuret, 2016. Character level language model based on http://karpathy.github.io/2015/05/21/rnn-effectiveness.
# opts=(:keepstate,false)(:lr,4.0)(:atype,"KnetArray{Float32}")(:winit,0.3)(:savefile,nothing)(:loadfile,nothing)(:generate,0)(:bestfile,nothing)(:gclip,3.0)(:embedding,256)(:hidden,256)(:epochs,3)(:decay,0.9)(:gcheck,0)(:seqlength,100)(:seed,42)(:batchsize,128)(:datafiles,Any["data/10.txt"])(:fast,true)
# INFO: Chars read: [("10.txt",425808)]
# INFO: 87 unique chars.
#   1.388652 seconds (1.73 M allocations: 196.686 MB, 2.04% gc time)
# (:epoch,0,:loss,6.1075509900258)
#   3.940298 seconds (6.06 M allocations: 373.166 MB, 2.06% gc time)
#   3.935995 seconds (6.06 M allocations: 373.244 MB, 2.07% gc time)
#   3.934983 seconds (6.06 M allocations: 373.245 MB, 2.08% gc time)
#   1.390374 seconds (1.73 M allocations: 196.820 MB, 2.11% gc time)
# (:epoch,3,:loss,1.860654126576015)



### SAMPLE RUN 31136d5+ Wed Sep 14 17:51:44 EEST 2016: using vcat(x,h) and vcat(w...)
### optimized learning parameters: winit=0.3, lr=4.0, gclip=3.0

# charlm.jl (c) Emre Yolcu, Deniz Yuret, 2016. Character level language model based on http://karpathy.github.io/2015/05/21/rnn-effectiveness.
# opts=(:keepstate,false)(:lr,4.0)(:atype,"KnetArray{Float32}")(:winit,0.3)(:savefile,nothing)(:loadfile,nothing)(:generate,0)(:bestfile,nothing)(:gclip,3.0)(:embedding,256)(:hidden,256)(:epochs,3)(:decay,0.9)(:gcheck,0)(:seqlength,100)(:seed,42)(:batchsize,128)(:datafiles,Any["10.txt"])(:fast,true)
# INFO: Chars read: [("10.txt",425808)]
# INFO: 87 unique chars.
#   1.596959 seconds (1.79 M allocations: 197.432 MB, 2.56% gc time)
# (:epoch,0,:loss,5.541976199042528)
#   4.421566 seconds (6.23 M allocations: 375.843 MB, 2.54% gc time)
#   4.418540 seconds (6.25 M allocations: 376.058 MB, 2.51% gc time)
#   4.402317 seconds (6.26 M allocations: 376.297 MB, 2.66% gc time)
#   1.594677 seconds (1.81 M allocations: 197.737 MB, 2.64% gc time)
# (:epoch,3,:loss,1.8484550957572192)


### SAMPLE RUN 80503e7+ Wed Sep 14 17:35:36 EEST 2016: using vcat(x,h)
#
# charlm.jl (c) Emre Yolcu, Deniz Yuret, 2016. Character level language model based on http://karpathy.github.io/2015/05/21/rnn-effectiveness.
# opts=(:keepstate,false)(:lr,1.0)(:atype,"KnetArray{Float32}")(:savefile,nothing)(:loadfile,nothing)(:generate,0)(:bestfile,nothing)(:embedding,256)(:gclip,5.0)(:hidden,256)(:epochs,3)(:decay,0.9)(:gcheck,0)(:seqlength,100)(:seed,42)(:batchsize,128)(:datafiles,Any["10.txt"])(:fast,true)
# INFO: Chars read: [("10.txt",425808)]
# INFO: 87 unique chars.
#   1.930180 seconds (1.95 M allocations: 213.741 MB, 1.82% gc time)
# (:epoch,0,:loss,4.462641664662756)
#   4.968101 seconds (7.47 M allocations: 454.259 MB, 2.24% gc time)
#   4.963733 seconds (7.47 M allocations: 454.363 MB, 2.26% gc time)
#   4.967413 seconds (7.45 M allocations: 454.024 MB, 2.14% gc time)
#   1.945658 seconds (1.98 M allocations: 214.183 MB, 2.02% gc time)
# (:epoch,3,:loss,3.2389672966290237)


### SAMPLE RUN 65f57ff+ Wed Sep 14 10:02:30 EEST 2016: separate x, h, w, b
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


### SAMPLE OUTPUT (with head -10000 100.txt): first version
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

