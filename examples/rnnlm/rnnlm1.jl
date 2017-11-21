for p in ("ArgParse","JLD","Knet")
    if Pkg.installed(p) == nothing; Pkg.add(p); end
end

module RNNLM; using ArgParse,JLD,Knet
using Knet: sigm_dot, tanh_dot
macro msg(_x) :(if logging>0; join(STDERR,[Dates.format(now(),"HH:MM:SS"), $(esc(_x)),'\n'],' '); end) end
macro log(_x) :(@msg($(string(_x))); $(esc(_x))) end

# sequence[t]::Vector{Int} minibatch of tokens
function rnnlm(model, sequence; pdrop=0, rat=nothing) # 2:1830 1:2585
    T = length(sequence)
    N = nlayers(model)
    batch = length(sequence[1])
    state = initstate(model,batch)
    total = count = 0
    for t in 1:(T-1)
        input = Wm(model)[:,sequence[t]]
        for n in 1:N
            w,b,h,c = Wh(model,n),bh(model,n),hdd(state,n),cll(state,n)
            input = dropout(input,pdrop)
            (h,c) = lstm(w,b,h,c,input)
            input = h
            state[2n-1] = h
            state[2n] = c
        end
        input = dropout(input,pdrop)
        logp0 = Wy(model) * input .+ by(model)
        logp1 = logp(logp0,1)
        golds = sequence[t+1]
        index = golds + size(logp1,1)*(0:(length(golds)-1))
        index = index[golds .!= PAD]
        logp2 = logp1[index]
        total += sum(logp2)
        count += length(index)
    end
    if rat != nothing; rat[1]=total; rat[2]=count; end
    # return -total / count # per token loss: scale does not depend on sequence length or minibatch
    # return -total / batch # per sequence loss: does not depend on minibatch, larger loss for longer seq
    return -total 	    # total loss: longer sequences and larger minibatches have higher loss
end

function perplexity(model, data)
    rat=zeros(2); tot=zeros(2)
    for sequence in data
        rnnlm(model, sequence; rat=rat)
        tot += rat
    end
    return exp(-tot[1] / tot[2])
end

function lstm(weight,bias,hidden,cell,input)            # 2:991  1:992:1617 (id:forw:back)
    gates   = weight * vcat(hidden, input) .+ bias      # 2:312  1:434:499 (43+381+75) (cat+mmul+badd)
    h       = size(hidden,1)                            # 
    forget  = sigm_dot(gates[1:h,:])                        # 2:134  1:98:99  (62+37) (index+sigm)
    ingate  = sigm_dot(gates[1+h:2h,:])                     # 2:99   1:73:123 (77+46)
    outgate = sigm_dot(gates[1+2h:3h,:])                    # 2:113  1:66:124 (87+37)
    change  = tanh_dot(gates[1+3h:4h,:])                    # 2:94   1:51:179 (130+49) replace end with 4h?
    cell    = cell .* forget + ingate .* change         # 2:137  1:106:202 (104+93+5) (bmul+bmul+add)
    hidden  = outgate .* tanh_dot(cell)                     # 2:100  1:69:194 (73+121) (tanh+bmul)
    return (hidden,cell)
end

nlayers(model)=div(length(model)-3,2)
Wm(model)=model[1]
Wx(model,n)=nothing
Wh(model,n)=model[2n]
bh(model,n)=model[2n+1]
Wy(model)=model[end-1]
by(model)=model[end]
hdd(state,n)=state[2n-1]
cll(state,n)=state[2n]

function initmodel(atype, hidden, vocab, embed)
    init(d...)=atype(xavier(Float32,d...))
    bias(d...)=atype(zeros(Float32,d...))
    N = length(hidden)
    model = Array{Any}(2N+3)
    model[1] = init(embed,vocab+1) # Wm
    X = embed
    for n = 1:N
        H = hidden[n]
        model[2n]   = init(4H,H+X) # Wh
        model[2n+1] = bias(4H,1) # bh
        model[2n+1][1:H] = 1     # forget gate bias = 1
        X = H
    end
    model[2N+2] = init(vocab,hidden[end]) # Wy
    model[2N+3] = bias(vocab,1)           # by
    return model
end

let blank = nothing; global initstate; using AutoGrad
function initstate(model, batch)
    N = nlayers(model)
    state = Array{Any}(2N)
    for n = 1:N
        bias = AutoGrad.getval(bh(model,n))
        hidden = div(length(bias),4)
        if typeof(blank)!=typeof(bias) || size(blank)!=(hidden,batch)
            blank = fill!(similar(bias, hidden, batch),0)
        end
        state[2n-1] = state[2n] = blank
    end
    return state
end
end

# initoptim creates optimization parameters for each numeric weight
# array in the model.  This should work for a model consisting of any
# combination of tuple/array/dict.
initoptim{T<:Number}(::KnetArray{T},otype)=eval(parse(otype))
initoptim{T<:Number}(::Array{T},otype)=eval(parse(otype))
initoptim(a::Associative,otype)=Dict(k=>initoptim(v,otype) for (k,v) in a) 
initoptim(a,otype)=map(x->initoptim(x,otype), a)

# Q: charlm minibatch style or sort sentences? => sort
# Q: padding or masking? => pad
# Q: initial state zero or learnt? => zero
# Q: partial batches => pad to full size

function minibatch(data, B)
    data = sort(data, by=length)
    D = length(data)
    O = ceil(Int, D/B)
    output = Array{Any}(O)
    for o in 1:O
        d = min(length(data), o*B) # idx of longest seq
        T = length(data[d])+1   # +1 for final EOS
        sbatch = Array{Any}(T+1) # +1 for initial EOS
        for t in 0:T
            wbatch=Array{Int32}(B)
            for b in 1:B
                d = (o-1)*B+b
                n = (d > length(data) ? 0 : length(data[d]))
                wbatch[b] = (d > length(data) ? PAD : t==0 ? EOS : t<=n ? data[d][t] : t==n+1 ? EOS : PAD)
            end
            sbatch[t+1] = wbatch
        end
        output[o] = sbatch
    end
    return output
end

function loaddata(files...; vocab=nothing)
    global EOS = 1 # use as both SOS in input and EOS in output.
    if isempty(files); files = mikolovptb(); end
    if any(!isfile(f) for f in files); error("File not found"); end
    if vocab == nothing; vocab = Dict{String,Int32}("<s>"=>EOS); end
    data = Any[]
    for f in files
        d = Any[]; nw = ns = 0
        for l in eachline(f)
            s = Int32[]; ns+=1
            for w in split(l)
                push!(s, get!(vocab, w, 1+length(vocab))); nw+=1
            end
            push!(d, s)
        end
        push!(data, d)
        @msg("$f: $ns sentences, $nw words, vocab=$(length(vocab))")
    end
    global PAD = length(vocab)+1 # set this to vocab+1, never assign any probability!
    return (data, vocab)
end

function mikolovptb()
    files = [ Knet.dir("data","ptb.$x.txt") for x in ("train","valid","test") ]
    if any(!isfile(f) for f in files)
        tgz = Knet.dir("data","simple-examples.tgz")
        if !isfile(tgz)
            url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
            download(url,tgz)
        end
        run(`tar --strip-components 3 -C $(Knet.dir("data")) -xzf $tgz ./simple-examples/data/ptb.train.txt ./simple-examples/data/ptb.valid.txt ./simple-examples/data/ptb.test.txt`)
    end
    return files
end

rnnlmgrad = grad(rnnlm)

function bptt(model, data, optim; pdrop=0)
    for sequence in data
        grads = rnnlmgrad(model, sequence; pdrop=pdrop)
        update!(model, grads, optim)
    end
end

function main(args=ARGS)
    global model, text, data, tok2int, o
    s = ArgParseSettings()
    s.description="rnnlm.jl: LSTM language model\n"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for early stop, others for test.")
        ("--loadfile"; help="Initialize model from file")
        ("--savefile"; help="Save final model to file")
        ("--bestfile"; help="Save best model to file")
        # ("--generate"; arg_type=Int; default=0; help="If non-zero generate given number of tokens.")
        ("--epochs"; arg_type=Int; default=1; help="Number of epochs for training.")
        ("--hidden"; nargs='+'; arg_type=Int; default=[256]; help="Sizes of one or more LSTM layers.")
        ("--embed"; arg_type=Int; default=128; help="Size of the embedding vector.")
        ("--batchsize"; arg_type=Int; default=64; help="Number of sequences to train on in parallel.")
        ("--optimization"; default="Adam()"; help="Optimization algorithm and parameters.")
        ("--dropout"; arg_type=Float64; default=0.0; help="Dropout probability.")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--logging"; arg_type=Int; default=1; help="display progress messages")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    global logging = o[:logging]
    @msg(string(s.description,"opts=",[(k,v) for (k,v) in o]...))
    if o[:seed] > 0; setseed(o[:seed]); end
    atype = eval(parse(o[:atype]))
    global text,vocab; (text,vocab) = loaddata(o[:datafiles]...)
    global data = map(t->minibatch(t, o[:batchsize]), text)
    global model = initmodel(atype, o[:hidden], length(vocab), o[:embed])
    report(ep)=(l=Float32[perplexity(model,d) for d in data];@msg((:epoch,ep,:perp,l...));l)
    if length(data) > 1; devset=2; else devset=1; end
    if !o[:fast]; @log losses = report(0); devbest = losses[devset]; end
    global optim = initoptim(model,o[:optimization])
    Knet.knetgc(); gc() # TODO: fix this otherwise curand cannot initialize no memory left!
    for epoch=1:o[:epochs]
        @log bptt(model, data[1], optim; pdrop=o[:dropout])
        if o[:fast]; continue; end
        @log losses = report(epoch)
        if o[:bestfile] != nothing && losses[devset] < devbest
            devbest = losses[devset]
            @log save(o[:bestfile], "model", model, "vocab", vocab)
        end
        if o[:gcheck] > 0
            gradcheck(rnnlm, model, rand(data[1]); gcheck=o[:gcheck], verbose=true)
        end
    end
    if o[:savefile] != nothing
        @log save(o[:savefile], "model", model, "vocab", vocab)
    end
    return model
end


# This allows both non-interactive (shell command) and interactive calls like:
# $ julia rnnlm.jl --epochs 10
# julia> RNNLM.main("--epochs 10")
if VERSION >= v"0.5.0-dev+7720"
    if basename(PROGRAM_FILE)==basename(@__FILE__); main(ARGS); end
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end  # module
