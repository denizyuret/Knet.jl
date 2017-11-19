# autograd fixes
# autograd use UngetIndex more aggresively (grad option)
# change UngetIndex to have array of indices rather than a single index (for dicts)
# knetarray Array{Int} indexing, transfer ints to gpu using KnetArray(::Array{Int})
# a single embed array with a single pointer is always going to be more efficient to pass around, if we can use it without constructing large gradients.
# in that case no need for concat, just indexing.
# can we traverse types and box their elements for autograd? check out deepcopy code again.
# add tests for new karray indexing ops
# fix charlm after finding optimum design.
# test repeated indices for getcols or getrows.

# (m,o,s)=main()
# @benchmark main(model=$m,opts=$o,sequence=$s,mode=0)

# mode=0  mode=1   mode=2   version (time in ms with default args)
# 86.367  156.640  160.112  0ddef27 2017-03-09 s2s benchmark added
# 30.134   90.061   93.959  a164664 2017-03-10 s2s with indexing for embedding
# 36.483   79.906   83.476  275a131 2017-03-14 s2s lstm output collected

using Knet,AutoGrad,BenchmarkTools

function main(;
              iters=1,
              vocab=10000,
              batch=128,
              hidden=128,
              seqlen=20,
              atype=KnetArray{Float32},
              otype=Adam,
              model=initmodel(hidden, vocab, atype),
              opts=optimizers(model,otype),
              sequence=randseq(vocab,batch,seqlen),
              mode=0,
              # dropout, gclip, layers, embedsize
              )
    if mode == 0
        for i in 1:iters
            s2s(model, sequence, sequence)
        end
    elseif mode == 1
        for i in 1:iters
            s2sgrad(model, sequence, sequence)
        end
    elseif mode == 2
        for i in 1:iters
            grads = s2sgrad(model, sequence, sequence)
            update!(model, grads, opts)
        end
    else
        error("mode=$mode")
    end
    return (model, opts, sequence)
end

function initmodel(H, V, atype)
    init(d...)=atype(xavier(d...))
    model = Dict{Symbol,Any}()
    model[:state0] = [ init(1,H), init(1,H) ]
    model[:embed1] = init(V,H)
    model[:encode] = [ init(2H,4H), init(1,4H) ]
    model[:embed2] = init(V,H)
    model[:decode] = [ init(2H,4H), init(1,4H) ]
    model[:output] = [ init(H,V), init(1,V) ]
    return model
end

function randseq(V,B,T)
    s = Vector{Vector{Int}}()
    for t in 1:T
        push!(s, rand(2:V,B))
    end
    return s
end

# trying to do logp in one shot increases mode=0 from 30 to 36 ms
function s2s(model, inputs, outputs)             # 
    state = initstate(inputs[1], model[:state0]) # 14
    for input in reverse(inputs)
        # input = model[:embed1][input,:]
        input = lstm_input(model[:embed1], input) # 85
        state = lstm(model[:encode], state, input) # 723
    end
    EOS = ones(Int, length(outputs[1]))
    # input = model[:embed2][EOS,:]
    input = lstm_input(model[:embed2], EOS) # 3
    preds = []
    sumlogp = 0
    for output in outputs
        state = lstm(model[:decode], state, input) # 702
        push!(preds, state[1])
        # ypred = predict(model[:output], state[1])
        # sumlogp += logprob(output, ypred)
        # input = model[:embed2][output,:]
        input = lstm_input(model[:embed2],output) # 61
    end
    state = lstm(model[:decode], state, input) # 30
    push!(preds, state[1])
    # ypred = predict(model[:output], state[1])
    # sumlogp += logprob(EOS, ypred)
    gold = vcat(outputs..., EOS) # 1
    sumlogp = lstm_output(model[:output], preds, gold) # 2441
    return -sumlogp
end

s2sgrad = grad(s2s)

function lstm_output(param, preds, gold)
    pred1 = vcat(preds...) # 46
    pred2 = pred1 * param[1] # 242
    pred3 = pred2 .+ param[2] # 145
    sumlogp = logprob(gold, pred3) # 2006
    return sumlogp
end

function lstm_output1(param, preds, gold, grads)
    # compute gradient wrt param
end

function lstm_output2(param, preds, gold, grads)
    # compute gradient wrt preds
end

function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = similar(output)
    @inbounds for i=1:length(output)
        index[i] = i + (output[i]-1)*nrows
    end
    o1 = logp(ypred,2)     # 1999
    o2 = o1[index]         # 4
    o3 = sum(o2)           # 2
    return o3
end

function lstm_input(param, input)
    p = param[input,:]     # 118
    return p
end

function lstm_input_back(param, input, grads)
    dparam = zeros(param)  # 157
    dparam[input,:]=grads  # 121
    return dparam
end

@primitive lstm_input(param,input),grads lstm_input_back(param,input,grads)

function lstm(param, state, input)
    weight,bias = param
    hidden,cell = state
    h       = size(hidden,2)
    gates   = hcat(input,hidden) * weight .+ bias
    forget  = sigm(gates[:,1:h])
    ingate  = sigm(gates[:,1+h:2h])
    outgate = sigm(gates[:,1+2h:3h])
    change  = tanh(gates[:,1+3h:4h])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

function predict(param, input)
    o1 = input * param[1]
    o2 = o1 .+ param[2]
    return o2
end

function initstate(idx, state0)
    h,c = state0
    h = h .+ fill!(similar(AutoGrad.getval(h), length(idx), length(h)), 0)
    c = c .+ fill!(similar(AutoGrad.getval(c), length(idx), length(c)), 0)
    return (h,c)
end

function _s2s(model, inputs, outputs)
    state = initstate(inputs[1], model[:state0])
    for input in reverse(inputs)
        input = model[:embed1][input,:]
        state = lstm(model[:encode], state, input)
    end
    EOS = ones(Int, length(outputs[1]))
    input = model[:embed2][EOS,:]
    sumlogp = 0
    for output in outputs
        state = lstm(model[:decode], state, input)
        ypred = predict(model[:output], state[1])
        sumlogp += logprob(output, ypred)
        input = model[:embed2][output,:]
    end
    state = lstm(model[:decode], state, input)
    ypred = predict(model[:output], state[1])
    sumlogp += logprob(EOS, ypred)
    return -sumlogp
end

nothing
# (m,o,s) = main()
# b = @benchmark main(model=$m,opts=$o,sequence=$s,mode=2)
# display(b)
# println()

#=
Profiling results:

Forward (mode=0):

=#
