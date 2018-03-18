for p in ("ArgParse","JLD","Knet")
    Pkg.installed(p) == nothing && Pkg.add(p)
end


"""

This example implements an LSTM network for training and testing
character-level language models inspired by ["The Unreasonable
Effectiveness of Recurrent Neural
Networks"](http://karpathy.github.io/2015/05/21/rnn-effectiveness)
from Andrej Karpathy's blog.  The model can be trained with different
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

* `julia charlm.jl --load foo.jld --generate 1000 --sresult generated.txt`:
  generates 1000 characters from the model in foo.jld and saves it to
  generated.txt.


* `julia charlm.jl --help`: describes all available options.

"""
module CharLM
using Knet,AutoGrad,ArgParse,JLD

# LSTM implementation with a single matrix multiplication with
# instances in rows rather than columns.  Julia is column major, so
# horizontal concatenation and column based slicing are contiguous and
# more efficient compared to vertical concatenation and row
# slicing. In this implementation I wanted to perform a single matrix
# multiplication for all four gates rather than four (or eight)
# separate matrix multiplications for performance. Thus I concatenate
# the input and the hidden, then slice out the four gates.  Both
# operations are more efficient if instances are in rows rather than
# columns.

function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm.(gates[:,1:hsize])
    ingate  = sigm.(gates[:,1+hsize:2hsize])
    outgate = sigm.(gates[:,1+2hsize:3hsize])
    change  = tanh.(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh.(cell)
    return (hidden,cell)
end

# generate model parameters for k=1:length(hidden) lstm layers
# instances are in rows, vectors are row vectors
# model[2k-1]: weight matrix for the k'th lstm layer
# model[2k]: bias vector for the k'th lstm layer
# model[end-2]: embedding matrix
# model[end-1,end]: weight and bias for final prediction
function initmodel(atype, hidden, vocab, embed)
    init(d...)=atype(xavier(d...))
    bias(d...)=atype(zeros(d...))
    model = Array{Any}(2*length(hidden)+3)
    X = embed
    for k = 1:length(hidden)
        H = hidden[k]
        model[2k-1] = init(X+H, 4H)
        model[2k] = bias(1, 4H)
        model[2k][1:H] = 1 # forget gate bias = 1
        X = H
    end
    model[end-2] = init(vocab,embed)
    model[end-1] = init(hidden[end],vocab)
    model[end] = bias(1,vocab)
    return model
end

# state[2k-1]: hidden for the k'th lstm layer
# state[2k]: cell for the k'th lstm layer
let blank = nothing; global initstate
function initstate(model, batch)
    nlayers = div(length(model)-3,2)
    state = Array{Any}(2*nlayers)
    for k = 1:nlayers
        bias = model[2k]
        hidden = div(length(bias),4)
        if typeof(blank)!=typeof(bias) || size(blank)!=(batch,hidden)
            blank = fill!(similar(bias, batch, hidden),0)
        end
        state[2k-1] = state[2k] = blank
    end
    return state
end
end

# initoptim creates optimization parameters for each numeric weight
# array in the model.  This should work for a model consisting of any
# combination of tuple/array/dict.
initoptim{T<:Number}(::KnetArray{T},otype)=eval(parse(otype))
initoptim{T<:Number}(::Array{T},otype)=eval(parse(otype))
# TODO: This breaks Julia4 parser:
# initoptim(a::Associative,otype)=Dict(k=>initoptim(v,otype) for (k,v) in a)
initoptim(a,otype)=map(x->initoptim(x,otype), a)

# input: Dense token-minibatch input
# returns (B,H) hidden output and newstate
function predict(model, state, input; pdrop=0)
    nlayers = div(length(model)-3,2)
    newstate = similar(state)
    for k = 1:nlayers
        input = dropout(input, pdrop)
        (newstate[2k-1],newstate[2k]) = lstm(model[2k-1],model[2k],state[2k-1],state[2k],input)
        input = newstate[2k-1]
    end
    return input,newstate
end

function generate(model, tok2int, nchar)
    int2tok = Array{Char}(length(tok2int))
    for (k,v) in tok2int; int2tok[v] = k; end
    input = tok2int[' ']
    state = initstate(model, 1)
    # Open file for saving
    if o[:sresult] != nothing
        f = open(o[:sresult],"w")
    end
    for t in 1:nchar
        embed = model[end-2][[input],:]
        ypred,state = predict(model,state,embed)
        ypred = ypred * model[end-1] .+ model[end]
        input = sample(exp.(logp(ypred)))
        print(int2tok[input])
        # Save character to file
        if o[:sresult] != nothing
            write(f, int2tok[input])
        end
    end
    # Close file if opened
    if o[:sresult] != nothing
        close(f)
    end
    println()
end

# sequence[t]: Vector{Int} token-minibatch input at time t
function loss(model, state, sequence, range=1:length(sequence)-1; newstate=nothing, pdrop=0)
    preds = []
    for t in range
        input = model[end-2][sequence[t],:]
        pred,state = predict(model,state,input; pdrop=pdrop)
        push!(preds,pred)
    end
    if newstate != nothing
        copy!(newstate, map(AutoGrad.getval,state))
    end
    pred0 = vcat(preds...)
    pred1 = dropout(pred0,pdrop)
    pred2 = pred1 * model[end-1]
    pred3 = pred2 .+ model[end]
    logp1 = logp(pred3,2)
    nrows,ncols = size(pred3)
    golds = vcat(sequence[range[1]+1:range[end]+1]...)
    index = similar(golds)
    @inbounds for i=1:length(golds)
        index[i] = i + (golds[i]-1)*nrows
    end
    # pred3 = Array(pred3) #TODO: FIX BUGGY REDUCTION CODE FOR KNETARRAY IF PRED3 TOO BIG
    logp2 = logp1[index]
    logp3 = sum(logp2)
    return -logp3 / length(golds)
end

lossgradient = grad(loss)

function avgloss(model, sequence, S)
    T = length(sequence)
    B = length(sequence[1])
    state = initstate(model, B)
    total = count = 0
    for i in 1:S:T-1
        j = min(i+S-1,T-1)
        n = j-i+1
        total += n * loss(model, state, sequence, i:j; newstate=state)
        count += n
    end
    return total / count
end

function bptt(model, sequence, optim, S; pdrop=0)
    T = length(sequence)
    B = length(sequence[1])
    state = initstate(model, B)
    for i in 1:S:T-1
        j = min(i+S-1,T-1)
        grads = lossgradient(model, state, sequence, i:j; newstate=state, pdrop=pdrop)
        update!(model, grads, optim)
    end
end

# It turns out convergence is much faster if we keep bptt length small
# initially. So we take user's o[:seqlength] as the upper limit and
# bound the bptt length by the epoch number.
function train!(model, data, tok2int, o)
    global optim = initoptim(model,o[:optimization])
    if o[:fast]
        for epoch=1:o[:epochs]
            bptt(model, data[1], optim, min(epoch,o[:seqlength]); pdrop=o[:dropout])
        end
        return
    end
    report(ep)=(l=map(d->avgloss(model,d,100), data);println((:epoch,ep,:loss,l...));l)
    @time losses = report(0)
    devset = ifelse(length(data) > 1, 2, 1)
    devlast = devbest = losses[devset]
    for epoch=1:o[:epochs]
        @time bptt(model, data[1], optim, min(epoch,o[:seqlength]); pdrop=o[:dropout])
        @time losses = report(epoch)
        if o[:gcheck] > 0
            gradcheck(loss, model, data[1], 1:min(o[:seqlength],length(data[1])-1); gcheck=o[:gcheck], verbose=true)
        end
        devloss = losses[devset]
        if devloss < devbest
            devbest = devloss
            if o[:bestfile] != nothing
                info("Saving best model to $(o[:bestfile])")
                save(o[:bestfile], "model", model, "vocab", tok2int)
            end
        end
        devlast = devloss
    end
    if o[:savefile] != nothing
        info("Saving final model to $(o[:savefile])")
        save(o[:savefile], "model", model, "vocab", tok2int)
    end
end

function minibatch(chars, tok2int, batch_size)
    chars = collect(chars)
    nbatch = div(length(chars), batch_size)
    data = [ zeros(Int,batch_size) for i=1:nbatch ]
    for n = 1:nbatch
        for b = 1:batch_size
            char = chars[(b-1)*nbatch + n]
            data[n][b] = tok2int[char]
        end
    end
    return data
end

function sample(p)
    p = convert(Array,p)
    r = rand()
    for c = 1:length(p)
        r -= p[c]
        r < 0 && return c
    end
end

function main(args=ARGS)
    global model, text, data, tok2int, o
    s = ArgParseSettings()
    s.description="charlm.jl: Character level language model based on http://karpathy.github.io/2015/05/21/rnn-effectiveness. (c) Emre Yolcu, Deniz Yuret, 2017."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
        ("--loadfile"; help="Initialize model from file")
        ("--savefile"; help="Save final model to file")
        ("--bestfile"; help="Save best model to file")
        ("--generate"; arg_type=Int; default=0; help="If non-zero generate given number of characters.")
        ("--epochs"; arg_type=Int; default=1; help="Number of epochs for training.")
        ("--hidden"; nargs='+'; arg_type=Int; default=[334]; help="Sizes of one or more LSTM layers.")
        ("--embed"; arg_type=Int; default=168; help="Size of the embedding vector.")
        ("--batchsize"; arg_type=Int; default=256; help="Number of sequences to train on in parallel.")
        ("--seqlength"; arg_type=Int; default=100; help="Maximum number of steps to unroll the network for bptt. Initial epochs will use the epoch number as bptt length for faster convergence.")
        ("--optimization"; default="Adam()"; help="Optimization algorithm and parameters.")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--dropout"; arg_type=Float64; default=0.0; help="Dropout probability.")
        ("--sresult"; help = "Save generated text to file" )
    end
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    o = parse_args(args, s; as_symbols=true)
    if !o[:fast]
        println(s.description)
        println("opts=",[(k,v) for (k,v) in o]...)
    end
    o[:seed] > 0 && srand(o[:seed])
    o[:atype] = eval(parse(o[:atype]))

    # we initialize a model from loadfile, train using datafiles (both optional).
    # if the user specifies neither, train a model using the charlm.jl source code.
    isempty(o[:datafiles]) && o[:loadfile]==nothing && push!(o[:datafiles],@__FILE__) # shakespeare()

    # read text and report lengths
    text = map((@compat readstring), o[:datafiles])
    !isempty(text) && !o[:fast] && info("Chars read: $(map((f,c)->(basename(f),length(c)),o[:datafiles],text))")

    # tok2int (char_to_index) comes from the initial model if there is one, otherwise from the datafiles.
    # if there is an initial model make sure the data has no new vocab
    if o[:loadfile]==nothing
        tok2int = Dict{Char,Int}()
        for t in text, c in t; get!(tok2int, c, 1+length(tok2int)); end
        model = initmodel(o[:atype], o[:hidden], length(tok2int), o[:embed])
    else
        info("Loading model from $(o[:loadfile])")
        tok2int = load(o[:loadfile], "vocab")
        for t in text, c in t; haskey(tok2int, c) || error("Unknown char $c"); end
        model = map(p->convert(o[:atype],p), load(o[:loadfile], "model"))
    end
    !o[:fast] && info("$(length(tok2int)) unique chars.")
    if !isempty(text)
        data = map(t->minibatch(t, tok2int, o[:batchsize]), text)
        train!(model, data, tok2int, o)
    end
    if o[:generate] > 0
        generate(model, tok2int, o[:generate])
    end
    return model
end


# This allows both non-interactive (shell command) and interactive calls like:
# $ julia charlm.jl --epochs 10
# julia> CharLM.main("--epochs 10")
PROGRAM_FILE=="charlm.jl" && main(ARGS)

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
