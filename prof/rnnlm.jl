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
#  16.227  36.886  39.329   32b63d3 2017-03-25 32546 wps on aitest-gpu
# 249.359 505.947 552.944   32b63d3 2017-03-25 2315  wps on aitest-cpu

using Knet,AutoGrad,BenchmarkTools

function main(;
              mode=0,
              iters=1,
              vocab=10000,
              batch=64,
              embed=128,
              hidden=[256],
              seqlen=20,        # mikolov ptb avg is 21+eos
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
            grads = rnnlmgrad(model, state, sequence; pdrop=dropout) # 1:6223 forw:2585 back:3636
            update!(model, grads, optim) # 1:943
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
    
function lstm(weight,bias,hidden,cell,input)                    # 1:992:1617 (id:forw:back)
    gates   = hcat(input,hidden) * weight .+ bias               # 1:129:499 (43+381+75) (cat+mmul+badd)
    h       = size(hidden,2)                                    # 
    forget  = sigm(gates[:,1:h])                                # 1:98:99  (62+37) (index+sigm)
    ingate  = sigm(gates[:,1+h:2h])                             # 1:73:123 (77+46)
    outgate = sigm(gates[:,1+2h:3h])                            # 1:66:124 (87+37)
    change  = tanh(gates[:,1+3h:4h])                            # 1:51:179 (130+49) replace end with 4h?
    cell    = cell .* forget + ingate .* change                 # 1:106:202 (104+93+5) (bmul+bmul+add)
    hidden  = outgate .* tanh(cell)                             # 1:69:194 (73+121) (tanh+bmul)
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
function rnnlm(model, state, sequence, range=1:length(sequence)-1; newstate=nothing, pdrop=0) # 1:2585
    preds = []
    embed = model[end-2]
    for t in range
        input = embed[sequence[t],:]                            # 1:86:92
        pred,state = predict(model,state,input; pdrop=pdrop) 	# 1:999:1617
        push!(preds,pred)
    end
    if newstate != nothing
        copy!(newstate, map(AutoGrad.getval,state))
    end
    pred0 = vcat(preds...)                                      # 1:51:35
    pred1 = dropout(pred0,pdrop)
    pred2 = pred1 * model[end-1]                                # 1:277:1132
    pred3 = pred2 .+ model[end]                                 # 1:84:33
    logp1 = logp(pred3,2)                                       # 1:1067:673
    nrows,ncols = size(pred3)
    golds = vcat(sequence[range[1]+1:range[end]+1]...)
    index = similar(golds)
    @inbounds for i=1:length(golds)
        index[i] = i + (golds[i]-1)*nrows                       # 1:17
    end
    # pred3 = Array(pred3) #TODO: FIX BUGGY REDUCTION CODE FOR KNETARRAY IF PRED3 TOO BIG
    logp2 = logp1[index]
    logp3 = sum(logp2)
    return -logp3 / length(golds)
end

rnnlmgrad = grad(rnnlm)

nothing

# Forward pass profile (2585/7166 of total)
# 2585 ./<missing>:0; (::#kw##rnnlm)(::Array{Any,1}, ::#rnnlm, ::AutoGrad.R...
#  86   .../dyuret/knet/master/prof/rnnlm.jl:158; input = embed[sequence[t],:]
#  999  .../dyuret/knet/master/prof/rnnlm.jl:159; pred,state = predict(model,state,input; pdrop=pdrop)
#   992 .../dyuret/knet/master/prof/rnnlm.jl:147; (newstate[2k-1],newstate[2k]) = lstm(model[2k-1],model[2k],state[2k-1],state[2k],input)
#    505 .../dyuret/knet/master/prof/rnnlm.jl:129; gates   = hcat(input,hidden) * weight .+ bias
#    98  .../dyuret/knet/master/prof/rnnlm.jl:131; forget  = sigm(gates[:,1:hsize])
#    73  .../dyuret/knet/master/prof/rnnlm.jl:132; ingate  = sigm(gates[:,1+hsize:2hsize])
#    66  .../dyuret/knet/master/prof/rnnlm.jl:133; outgate = sigm(gates[:,1+2hsize:3hsize])
#    51  .../dyuret/knet/master/prof/rnnlm.jl:134; change  = tanh(gates[:,1+3hsize:end])
#    106 .../dyuret/knet/master/prof/rnnlm.jl:135; cell    = cell .* forget + ingate .* change
#    69  .../dyuret/knet/master/prof/rnnlm.jl:136; hidden  = outgate .* tanh(cell)
#  51   .../dyuret/knet/master/prof/rnnlm.jl:165; pred0 = vcat(preds...)
#  277  .../dyuret/knet/master/prof/rnnlm.jl:167; pred2 = pred1 * model[end-1]
#  84   .../dyuret/knet/master/prof/rnnlm.jl:168; pred3 = pred2 .+ model[end]
#  1067 .../dyuret/knet/master/prof/rnnlm.jl:169; logp1 = logp(pred3,2)
#   424 ...ret/.julia/v0.5/Knet/src/unary.jl:176; x1 = maximum(x,d...)
#   53  ...ret/.julia/v0.5/Knet/src/unary.jl:177; x2 = x .- x1
#   88  ...ret/.julia/v0.5/Knet/src/unary.jl:178; x3 = exp(x2)
#   414 ...ret/.julia/v0.5/Knet/src/unary.jl:179; x4 = sum(x3,d...)
#   1   ...ret/.julia/v0.5/Knet/src/unary.jl:180; x5 = log(x4)
#   85  ...ret/.julia/v0.5/Knet/src/unary.jl:181; x6 = x2 .- x5
#  17   .../dyuret/knet/master/prof/rnnlm.jl:174; index[i] = i + (golds[i]-1)*nrows

# Backward pass profile (3636/7166 of total)
# 3636 ...et/.julia/v0.5/AutoGrad/src/core.jl:40; (::AutoGrad.##gradfun#1#3{#rnnlm,Int64})(::Array{Any,1...
#  2948 core.jl:231 og = r.func(Grad{i},n.outgrad,r.value,r.args...;r.kwargs...)
#   1512 ./<missing>:0; *(::Type{AutoGrad.Grad{2}}, ::Knet.KnetArray{Float32...
#   679  ./<missing>:0; logp(::Type{AutoGrad.Grad{1}}, ::Knet.KnetArray{Floa...
#   286  ./<missing>:0; .*(::Type{AutoGrad.Grad{1}}, ::Knet.KnetArray{Float3...
#   109  ./<missing>:0; sigm(::Type{AutoGrad.Grad{1}}, ::Knet.KnetArray{Floa...
#   95   ./<missing>:0; tanh(::Type{AutoGrad.Grad{1}}, ::Knet.KnetArray{Floa...
#   92   ./<missing>:0; .+(::Type{AutoGrad.Grad{2}}, ::Knet.KnetArray{Float3...
#   60   ...AutoGrad/src/base/abstractarray.jl:85; cat(::Type{AutoGrad.Grad{2}}, ::Knet.KnetArray{Float...
#   13   ./<missing>:0; getindex(::Type{AutoGrad.Grad{1}}, ::Knet.KnetArray{...
#   13   ./essentials.jl:216; vector_any()
#  684  ...et/.julia/v0.5/AutoGrad/src/core.jl:233; backward_pass(::AutoGrad.Rec{Array{Any,1}}, ::AutoGra...
#   271 ...uret/.julia/v0.5/Knet/src/karray.jl:995; sum_outgrads(::Knet.KnetArray{Float32,2}, ::AutoGrad....
#   168 ...lia/v0.5/AutoGrad/src/interfaces.jl:71; sum_outgrads(::Void, ::AutoGrad.UngetIndex)
#   167 ...lia/v0.5/AutoGrad/src/interfaces.jl:92; sum_outgrads(::Array{Any,1}, ::AutoGrad.UngetIndex)
#   72  ...uret/.julia/v0.5/Knet/src/karray.jl:992; sum_outgrads(::Knet.KnetArray{Float32,2}, ::Knet.Knet...

# Update profile (943/7166 of total)
# 940 ...yuret/.julia/v0.5/Knet/src/update.jl:404; update!(wi,gi,pi)
#  31  ...uret/.julia/v0.5/Knet/src/update.jl:346; scale!(p.beta1, p.fstm)
#  53  ...uret/.julia/v0.5/Knet/src/update.jl:347; axpy!(1-p.beta1, g, p.fstm)
#  35  ...uret/.julia/v0.5/Knet/src/update.jl:348; scale!(p.beta2, p.scndm)
#  82  ...uret/.julia/v0.5/Knet/src/update.jl:349; axpy!(1-p.beta2, g .* g, p.scndm)
#  29  ...uret/.julia/v0.5/Knet/src/update.jl:350; fstm_corrected = p.fstm / (1 - p.beta1 ^ p.t)
#  27  ...uret/.julia/v0.5/Knet/src/update.jl:351; scndm_corrected = p.scndm / (1 - p.beta2 ^ p.t)
#  683 ...uret/.julia/v0.5/Knet/src/update.jl:352; axpy!(-p.lr, (fstm_corrected ./ (sqrt(scndm_corrected) + p.eps)), w)
