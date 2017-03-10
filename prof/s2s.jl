# autograd fixes
# autograd use UngetIndex more aggresively (grad option)
# change UngetIndex to have array of indices rather than a single index (for dicts)
# knetarray Array{Int} indexing, transfer ints to gpu using KnetArray(::Array{Int})
# a single embed array with a single pointer is always going to be more efficient to pass around, if we can use it without constructing large gradients.
# in that case no need for concat, just indexing.
# can we traverse types and box their elements for autograd? check out deepcopy code again.
# add tests for new karray indexing ops
# fix charlm after finding optimum design.

# mode=0  mode=1   mode=2   version (time in ms with default args)
# 86.367  156.640  160.112  0ddef27 2017-03-09 s2s benchmark added
# 30.134   90.061   93.959  a164664 2017-03-10 s2s with indexing for embedding

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
              opts=oparams(model,otype),
              sequence=randseq(vocab,batch,seqlen),
              mode=2,
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
    Knet.cudaDeviceSynchronize()
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

# This should work for any combination of tuple/array/dict
oparams{T<:Number}(::KnetArray{T},otype; o...)=otype(;o...)
oparams{T<:Number}(::Array{T},otype; o...)=otype(;o...)
oparams(a::Associative,otype; o...)=Dict(k=>oparams(v,otype;o...) for (k,v) in a)
oparams(a,otype; o...)=map(x->oparams(x,otype;o...), a)

function randseq(V,B,T)
    s = Vector{Vector{Int}}()
    for t in 1:T
        push!(s, rand(2:V,B))
    end
    return s
end

function s2s(model, inputs, outputs)
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

s2sgrad = grad(s2s)

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

function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = similar(output)
    @inbounds for i=1:length(output)
        index[i] = i + (output[i]-1)*nrows
    end
    o1 = logp(ypred,2)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
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

#=

let z=nothing; global onehotrows
function onehotrows(idx, embeddings)
    nrows,ncols = length(idx), size(embeddings,1)
    if z==nothing || size(z) != (nrows,ncols)
        info("alloc z")
        z = Array(Float32,nrows,ncols)
    end
    fill!(z,0)
    @inbounds for i=1:nrows
        z[i,idx[i]] = 1
    end
    oftype(AutoGrad.getval(embeddings),z)
end
end

let EOS=nothing; global eosmatrix
function eosmatrix(idx, embeddings)
    nrows,ncols = length(idx), size(embeddings,1)
    if EOS==nothing || size(EOS) != (nrows,ncols) || typeof(EOS) != typeof(AutoGrad.getval(embeddings))
        info("alloc eos")
        EOS = zeros(Float32,nrows,ncols)
        EOS[:,1] = 1
        EOS = oftype(AutoGrad.getval(embeddings), EOS)
    end
    return EOS
end
end

function logprob(param, state, output)
    pred = predict(param, state)
    sum(logp(pred,2) .* output)
end

function lstminput(input, embedding)
    input = onehotrows(input, embedding)
    input = input * embedding
    return input
end

function translate(model, str)
    state = model[:state0]
    for c in reverse(collect(str))
        input = onehotrows(tok2int[c], model[:embed1])
        input = input * model[:embed1]
        state = lstm(model[:encode], state, input)
    end
    input = eosmatrix(1, model[:embed2]) * model[:embed2]
    output = Char[]
    for i=1:100 #while true
        state = lstm(model[:decode], state, input)
        pred = predict(model[:output], state)
        i = indmax(Array(pred))
        i == 1 && break
        push!(output, int2tok[i])
        input = onehotrows(i, model[:embed2]) * model[:embed2]
    end
    String(output)
end

function readdata(file="/usr/share/dict/words")
    strings = map(chomp,readlines(file))
    global tok2int = Dict{Char,Int}()
    global int2tok = Vector{Char}()
    push!(int2tok,'\n'); tok2int['\n']=1 # We use '\n'=>1 as the EOS token
    sequences = Vector{Vector{Int}}()
    for w in strings
        s = Vector{Int}()
        for c in collect(w)
            if !haskey(tok2int,c)
                push!(int2tok,c)
                tok2int[c] = length(int2tok)
            end
            push!(s, tok2int[c])
        end
        push!(sequences, s)
    end
    return sequences
end

function minibatch(sequences, batchsize)
    table = Dict{Int,Vector{Vector{Int}}}()
    data = Any[]
    for s in sequences
        n = length(s)
        nsequences = get!(table, n, Any[])
        push!(nsequences, s)
        if length(nsequences) == batchsize
            push!(data, [[ nsequences[i][j] for i in 1:batchsize] for j in 1:n ])
            empty!(nsequences)
        end
    end
    return data
end

function train(model, data, opts)
    for sequence in data
        grads = s2sgrad(model, sequence, sequence)
        update!(model, grads, opts)
    end
end

function avgloss(model, data)
    sumloss = cntloss = 0
    for sequence in data
        tokens = (1 + length(sequence)) * length(sequence[1])
        sumloss += s2s(model, sequence, sequence)
        cntloss += tokens
    end
    return sumloss/cntloss
end

function icat(vectors, indices)
    v = vectors[1]
    n = length(v)
    matrix = similar(v, n, length(indices))
    pos = 1
    for i in indices
        v = vectors[i]
        copy!(matrix, pos, v, 1, n)
        pos += n
    end
    return matrix
end

using Knet: libknet8

for (T,F) in ((Float32,"icat_32"),(Float64,"icat_64")); @eval begin
    function icat(vectors::Vector{KnetArray{$T,1}}, indices::Vector{Int})
        rows = length(vectors[1])
        cols = length(indices)
        matrix = similar(vectors[1], rows, cols)
        ptrs = map(pointer, view(vectors,indices))
        ccall(($F,libknet8),Void,(Cint,Cint,Ptr{Ptr{$T}},Ptr{$T}),rows,cols,ptrs,matrix)
        return matrix
    end
end; end

# This is already efficient, no copying, just views and axpy!
function icatback(vectors, indices, dmatrix)
    dvectors = Array(Any, length(vectors))
    fill!(dvectors, nothing)
    n = length(vectors[1])
    pos = 0
    for i in indices
        if dvectors[i] == nothing
            dvectors[i] = dmatrix[pos+1:pos+n]
        else
            axpy!(1, dmatrix[pos+1:pos+n], dvectors[i])
        end
        pos += n
    end
    return dvectors
end

@primitive  icat(x,i),dy  icatback(x,i,dy)

function irnn(param, state, input)
    relu(hcat(input,state) * param[1] .+ param[2])
end

function encoder1(model, state, input)
    input = model[2] * input
    state = relu(model[3] * state .+ model[4] * input .+ model[5])
    return state
end

function decoder1(model, state, input)
    input = model[6] * input
    state = relu(model[7] * state .+ model[8] * input .+ model[9])
    output = model[10] * state .+ model[11]
    return (state,output)
end


function _initmodel(H, V, B; atype=(gpu()>=0 ? KnetArray{Float32} : Array{Float32}))
    init(d...)=atype(xavier(d...))
    model = Array(Any,6)
    model[1] = [ init(B,H), init(B,H) ]
    model[2] = [ init(H) for i=1:V ]
    model[3] = [ init(2H,4H), init(1,4H) ]
    model[4] = [ init(H) for i=1:V ]
    model[5] = [ init(2H,4H), init(1,4H) ]
    model[6] = [ init(H,V), init(1,V) ]
    return model
end

function _s2s(model, inputs, outputs)
    state = model[1]
    for input in reverse(inputs)
        input = icat(model[2], input)'
        state = lstm(model[3], state, input)
    end
    sumlogp = 0
    batchsize = length(outputs[1])
    EOS = ones(Int,batchsize)
    input = icat(model[4], EOS)'
    for output in outputs
        state = lstm(model[5], state, input)
        sumlogp += logprob(model[6], state, output)
        input = icat(model[4], output)'
    end
    state = lstm(model[5], state, input)
    sumlogp += logprob(model[6], state, EOS)
    return -sumlogp / (batchsize * (1 + length(outputs)))
end

=#
nothing
