# Usage:
#
# include("rnnlm.jl")
# m,s,x,o = main(iters=0)
# for i=1:2
# gc(); @time main(model=m,state=s,sequence=x,optim=o,mode=0,iters=10)
# gc(); @time main(model=m,state=s,sequence=x,optim=o,mode=1,iters=10)
# gc(); @time main(model=m,state=s,sequence=x,optim=o,mode=2,iters=10)
# end
# gc(); println(@benchmark main(model=$m,state=$s,sequence=$x,optim=$o,mode=0))
# gc(); println(@benchmark main(model=$m,state=$s,sequence=$x,optim=$o,mode=1))
# gc(); println(@benchmark main(model=$m,state=$s,sequence=$x,optim=$o,mode=2))
# nothing

#  mode=0  mode=1  mode=2   notes (times in ms with default args)
#  16.227  36.886  39.329   32546 wps on aitest-gpu
# 249.359 505.947 552.944   2315  wps on aitest-cpu

using Knet,AutoGrad,BenchmarkTools

function main(;
              mode=0,
              iters=1,
              vocab=10000,
              batch=64,
              embed=128,
              hidden=[256],
              seqlen=20,        # mikolov ptb avg is 21
              otype="Adam()",
              atype=(gpu()>=0 ? KnetArray{Float32} : Array{Float32}),
              model=initmodel(atype, hidden, vocab, embed),
              state=initstate(model,batch),
              sequence=randseq(vocab,batch,seqlen),
              optim=initoptim(model,otype),
              dropout=0,
              # gclip
              )
    if mode == 0
        for i in 1:iters
            rnnlm(model, state, sequence; pdrop=dropout)
        end
    elseif mode == 1
        for i in 1:iters
            rnnlmgrad(model, state, sequence; pdrop=dropout)
        end
    elseif mode == 2
        for i in 1:iters
            grads = rnnlmgrad(model, state, sequence; pdrop=dropout)
            update!(model, grads, optim)
        end
    else
        error("mode=$mode")
    end
    return (model, state, sequence, optim)
end

# generate model parameters for k=1:length(hidden) lstm layers
# instances are in rows, vectors are row vectors
# model[2k-1]: weight matrix for the k'th lstm layer
# model[2k]: bias vector for the k'th lstm layer
# model[end-2]: embedding matrix
# model[end-1,end]: weight and bias for final prediction
function initmodel(atype, hidden, vocab, embed)
    init(d...)=atype(xavier(Float32,d...))
    bias(d...)=atype(zeros(Float32,d...))
    model = Array(Any, 2*length(hidden)+3)
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

# TODO: consider learning the initial state
# state[2k-1]: hidden for the k'th lstm layer
# state[2k]: cell for the k'th lstm layer
let blank = nothing; global initstate
function initstate(model, batch)
    nlayers = div(length(model)-3,2)
    state = Array(Any, 2*nlayers)
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
initoptim(a::Associative,otype)=Dict(k=>initoptim(v,otype) for (k,v) in a) 
initoptim(a,otype)=map(x->initoptim(x,otype), a)

# Create a random minibatch of sequences
function randseq(V,B,T)
    s = Vector{Vector{Int}}()
    for t in 1:T
        push!(s, rand(2:V,B))   # Using 1 for EOS
    end
    return s
end

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
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

# input: Matrix{Float} token-minibatch input
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

# sequence[t]: Vector{Int} token-minibatch input at time t
function rnnlm(model, state, sequence, range=1:length(sequence)-1; newstate=nothing, pdrop=0)
    preds = []
    embed = model[end-2]
    for t in range
        input = embed[sequence[t],:]
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

rnnlmgrad = grad(rnnlm)

nothing
