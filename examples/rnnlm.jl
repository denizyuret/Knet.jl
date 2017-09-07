for p in ("ArgParse","JLD","Knet")
    if Pkg.installed(p) == nothing; Pkg.add(p); end
end

module RNNLM
using ArgParse,JLD,Knet
using AutoGrad: getval
using Knet: sigm_dot, tanh_dot
logprint(x)=join(STDERR,[Dates.format(now(),"HH:MM:SS"),x,'\n'],' ')
macro run(i,x) :(if loglevel>=$i; $(esc(x)); end) end
macro msg(i,x) :(if loglevel>=$i; logprint($(esc(x))); end) end
macro log(i,x) :(if loglevel>=$i; logprint($(string(x))); end; $(esc(x))) end


function lstm(weight,bias,hidden,cell,input)            # 2:991  1:992:1617 (id:forw:back)
    gates   = weight * hidden .+ input .+ bias          # 2:312  1:434:499 (43+381+75) (cat+mmul+badd)
    h       = size(hidden,1)                            # 
    forget  = sigm_dot(gates[1:h,:])                        # 2:134  1:98:99  (62+37) (index+sigm)
    ingate  = sigm_dot(gates[1+h:2h,:])                     # 2:99   1:73:123 (77+46)
    outgate = sigm_dot(gates[1+2h:3h,:])                    # 2:113  1:66:124 (87+37)
    change  = tanh_dot(gates[1+3h:4h,:])                    # 2:94   1:51:179 (130+49) replace end with 4h?
    cell    = cell .* forget + ingate .* change         # 2:137  1:106:202 (104+93+5) (bmul+bmul+add)
    hidden  = outgate .* tanh_dot(cell)                     # 2:100  1:69:194 (73+121) (tanh+bmul)
    return (hidden,cell)
end

# sequence[t]::Vector{Int} minibatch of tokens
function rnnlm(model, state, sequence; pdrop=0, range=1:(length(sequence)-1), keepstate=nothing, stats=nothing)
    index = vcat(sequence[range]...)
    input = Wm(model)[:,index]                          # 2:15
    for n = 1:nlayers(model)
        input = dropout(input, pdrop)
        input = Wx(model,n) * input                     # 2:26
        w,b,h,c = Wh(model,n),bh(model,n),hdd(state,n),cll(state,n)
        output = []
        j1 = j2 = 0
        for t in range
            j1 = j2 + 1
            j2 = j1 + length(sequence[t]) - 1
            input_t = input[:,j1:j2]                    # 2:35
            (h,c) = lstm(w,b,h,c,input_t)               # 2:991
            push!(output,h)
        end
        if keepstate != nothing
            keepstate[2n-1] = getval(h)
            keepstate[2n] = getval(c)
        end
        input = hcat(output...)                         # 2:39
    end
    input = dropout(input,pdrop)
    input = Wy(model) * input .+ by(model)
    input = logp(input,1)                               # 2:354  1:1067:673
    golds = vcat(sequence[range+1]...)
    index = golds + size(input,1)*(0:(length(golds)-1))
    logp1 = input[index]
    total = sum(logp1)
    count = length(logp1)
    if stats != nothing
        stats[1]=total
        stats[2]=count
    end
    batch = length(sequence[1])
    # return -total / count # per token loss: scale does not depend on sequence length or minibatch
    return -total / batch # per sequence loss: does not depend on minibatch, larger loss for longer seq
    # return -total 	    # total loss: longer sequences and larger minibatches have higher loss
end

rnnlmgrad = grad(rnnlm)

# data[t][b] contains word[(b-1)*T+t]
function bptt(model, data, optim; pdrop=0, slen=20)
    T = length(data)
    B = length(data[1])
    state = initstate(model,B)
    @run 2 begin
        wnorm = zeros(length(model))
        gnorm = zeros(length(model))
        count = 0
    end
    for i = 1:slen:(T-1)
        j = i+slen-1
        if j >= T; break; end
        grads = rnnlmgrad(model, state, data; pdrop=pdrop, range=i:j, keepstate=state)
        update!(model, grads, optim)
        @run 2 begin
            gnorm += map(vecnorm,grads)
            wnorm += map(vecnorm,model)
            count += 1
        end
    end
    @msg 2 string("wnorm=",wnorm./count)
    @msg 2 string("gnorm=",gnorm./count)
end

function loss(model, data; slen=20)
    T = length(data)
    B = length(data[1])
    state = initstate(model,B)
    rat=zeros(2); tot=zeros(2)
    for i = 1:slen:(T-1)
        j = i+slen-1
        if j >= T; break; end
        rnnlm(model, state, data; stats=rat, range=i:j, keepstate=state)
        tot += rat
    end
    return (-tot[1],tot[2])
end

nlayers(model)=div(length(model)-3,3)
Wm(model)=model[1]
Wx(model,n)=model[3n-1]
Wh(model,n)=model[3n]
bh(model,n)=model[3n+1]
Wy(model)=model[end-1]
by(model)=model[end]
hdd(state,n)=state[2n-1]
cll(state,n)=state[2n]

function initmodel(atype, hidden, vocab, embed)
    init(d...)=atype(xavier(Float32,d...))
    bias(d...)=atype(zeros(Float32,d...))
    N = length(hidden)
    model = Array{Any}(3N+3)
    model[1] = init(embed,vocab) # Wm
    X = embed
    for n = 1:N
        H = hidden[n]
        model[3n-1] = init(4H,X) # Wx
        model[3n]   = init(4H,H) # Wh
        model[3n+1] = bias(4H,1) # bh
        model[3n+1][1:H] = 1     # forget gate bias = 1
        X = H
    end
    model[3N+2] = init(vocab,hidden[end]) # Wy
    model[3N+3] = bias(vocab,1)           # by
    return model
end

let blank = nothing; global initstate
function initstate(model, batch)
    N = nlayers(model)
    state = Array{Any}(2N)
    for n = 1:N
        bias = bh(model,n)
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

# Q: charlm flat minibatch style or sort sentences? => flat
# Q: padding or masking? => no need
# Q: initial state zero or learnt? => zero
# Q: partial batches => no need

function minibatch(data, B)
    T = div(length(data),B)
    batches = Array{Vector{Int32}}(T)
    for t = 1:T
        batch = Array{Int32}(B)
        for b = 1:B
            batch[b] = data[(b-1)*T+t]
        end
        batches[t] = batch
    end
    return batches
end

function initvocab()
    global EOS = Int32(1) # use as both SOS in input and EOS in output.
    Dict{String,Int32}("<s>"=>EOS)
end

function loaddata(file, vocab)
    data = Int32[EOS]; nw = ns = 0
    for l in eachline(file); ns+=1
        for w in split(l); nw+=1
            push!(data, get!(vocab, w, 1+length(vocab)))
        end
        push!(data,EOS)
    end
    @msg 1 "$file: $ns sentences, $nw words, vocab=$(length(vocab)), corpus=$(length(data))"
    return data
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

function main(args=ARGS)
    global model, text, data, tok2int, o
    s = ArgParseSettings()
    s.description="rnnlm.jl: LSTM language model\n"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for early stop, others for test. If not provided use mikolovptb files.")
        ("--loadfile"; help="Initialize model from file")
        ("--savefile"; help="Save final model to file")
        ("--bestfile"; help="Save best model to file")
        ("--epochs"; arg_type=Int; default=1; help="Number of epochs for training.")
        ("--hidden"; nargs='+'; arg_type=Int; default=[256]; help="Sizes of one or more LSTM layers.")
        ("--embed"; arg_type=Int; default=128; help="Size of the embedding vector.")
        ("--batchsize"; arg_type=Int; default=64; help="Number of sequences to train on in parallel.")
        ("--bptt"; arg_type=Int; default=20; help="Number of steps to unroll for bptt.")
        ("--optimization"; default="Adam()"; help="Optimization algorithm and parameters.")
        ("--dropout"; arg_type=Float64; default=0.0; help="Dropout probability.")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--loglevel"; arg_type=Int; default=1; help="display progress messages")
        # TODO: ("--generate"; arg_type=Int; default=0; help="If non-zero generate given number of tokens.")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    global loglevel = o[:loglevel]
    if o[:seed] > 0; setseed(o[:seed]); end
    if isempty(o[:datafiles]); o[:datafiles] = mikolovptb(); end
    @msg 1 string(s.description,"opts=",[(k,v) for (k,v) in o]...)
    global vocab = initvocab()
    global text = map(f->loaddata(f,vocab), o[:datafiles])
    global data = map(t->minibatch(t, o[:batchsize]), text)
    global model = initmodel(eval(parse(o[:atype])), o[:hidden], length(vocab), o[:embed])
    function report(ep)
        l = [ loss(model,d;slen=o[:bptt]) for d in data ]
        l1 = Float32[ exp(x[1]/x[2]) for x in l ]
        l2 = [ x[2] for x in l ]
        if ep==0; @msg 1 (:epoch,ep,:perp,l1...,:size,l2...)
        else; @msg 1 (:epoch,ep,:perp,l1...); end
        return l1
    end
    if length(data) > 1; devset=2; else devset=1; end
    if !o[:fast]; @log 1 (losses = report(0)); devbest = devlast = losses[devset]; end
    global optim = initoptim(model,o[:optimization])
    Knet.knetgc(); gc() # TODO: fix this otherwise curand cannot initialize no memory left!
    for epoch=1:o[:epochs]
        @log 1 bptt(model, data[1], optim; pdrop=o[:dropout], slen=o[:bptt])
        if o[:fast]; continue; end
        @log 1 (losses = report(epoch))
        if o[:bestfile] != nothing && losses[devset] < devbest
            devbest = losses[devset]
            @log 1 save(o[:bestfile], "model", model, "vocab", vocab)
        end
        # if epoch > 6 # losses[devset] > devlast && isa(optim[1], Sgd)
        #     for p in optim; p.lr /= 1.2; end; @msg 1 "lr=$(optim[1].lr)"
        # end
        devlast = losses[devset]
        if o[:gcheck] > 0
            gradcheck(rnnlm, model, initstate(model,o[:batchsize]), data[1]; gcheck=o[:gcheck], verbose=true, kwargs=[(:range,1:o[:bptt])])
        end
    end
    if o[:savefile] != nothing
        @log 1 save(o[:savefile], "model", model, "vocab", vocab)
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
