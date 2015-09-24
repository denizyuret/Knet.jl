import Base: start, next, done

type LMData; data; dict; batchsize; seqlen; epochsize; xbatch; ybatch; end

function LMData(fname::String; epoch=0, batch=20, seqlen=20, dict=Dict{Any,Int32}())
    data = Int32[]
    f = open(fname)
    for l in eachline(f)
        for w in split(l)
            push!(data, get!(dict, w, 1+length(dict)))
        end
        push!(data, get!(dict, "", 1+length(dict)))
    end
    # Compute an epoch size that is <= length(data) and an exact multiple of batch*seqlen
    ep = epoch
    (ep == 0 || ep > length(data)) && (ep = length(data))
    ep -= ep % (batch*seqlen)   # TODO: epoch should be number of sequences, or batch should be number of words
    ep != epoch && warn("Adjusting epoch size to $ep")
    x = [ speye(Float32, length(dict), batch) for i=1:seqlen ] # TODO: Ti=Int32
    y = [ speye(Float32, length(dict), batch) for i=1:seqlen ] # TODO: no need to alloc separate x/y, use same arrays
    LMData(data, dict, batch, seqlen, ep, x, y)
end

function next(d::LMData,n)                              # d.data is the whole corpus represented as a sequence of Int32's
    segsize = div(length(d.data), d.batchsize)          # we split it into d.batchsize roughly equal sized segments
    offset = div(n, d.batchsize)                        # this is how many words have been served so far from each segment
    for b = 1:d.batchsize
        idata = (b-1)*segsize + offset                  # the start of segment b is at 1+(b-1)*segsize
        for t = 1:d.seqlen
            d.ybatch[t].rowval[b] = d.data[idata+t]
            d.xbatch[t].rowval[b] = (idata+t == 1 ? d.dict[""] : d.data[idata+t-1])
        end
    end
    ((d.xbatch, d.ybatch), n + d.batchsize * d.seqlen)	# each call to next will deliver d.seqlen words from each of the d.batchsize segments
end

start(d::LMData)=(checkbatchsize(d); 0)
done(d::LMData,n)=(n>=d.epochsize)

function checkbatchsize(d::LMData) # dict length may increase if we are using shared dicts
    mx = size(d.xbatch[1], 1)
    nd = length(d.dict)
    nd == mx && return
    nd  < mx && error("Dictionary shrinkage")
    for x in d.xbatch; x.m = nd; end
    for y in d.ybatch; y.m = nd; end
end

dir = "simple-examples/data"
trn = LMData("$dir/ptb.train.txt")
dev = LMData("$dir/ptb.valid.txt"; dict=trn.dict)
tst = LMData("$dir/ptb.test.txt"; dict=trn.dict)

using KUnet, CUDArt
using KUnet: params

nh = 200
nw = length(dev.dict)
lr = 1.0
net = Net(Mmul(nh),LSTM(nh),LSTM(nh),Mmul(nw),Bias(),Soft(),SoftLoss())
setparam!(net, lr=lr)
setseed(42)

for ep=1:13
    ep > 4 && (lr /= 2; setparam!(net, lr=lr))
    @time (ltrn,w,g) = train(net, trn; gclip=10.0, keepstate=true)
    ptrn = exp(ltrn/trn.seqlen)
    @time ldev = test(net, dev)
    pdev = exp(ldev/dev.seqlen)
    @show (ep, pdev, ptrn, w, g)
end


### DEAD CODE:
# The input is a text file with word tokens, single sentence per line.
# 1. We can take the whole corpus as one sequence and use truncated BPTT.
# -- zaremba uses T=5 B=20 blocks
# -- this requires not resetting the hidden states.
# -- k1 and k2 are both 5 in Sutskever's truncated BPTT formulation.
# 2. We can take individual sentences as sequences and train normally.
# -- means a lot fewer updates (1 per sentence*minibatch)
# -- similar per epoch time expected with fewer updates
# -- don't need to change any code
# -- sort sentences by length to minimize padding
#
# Architecture:
# x,y both come in as sparse matrices size VxB
# Mmul(HxV) will convert to embedding, needs to handle sparse x
# Followed by 2xLSTM(H)
# Followed by Mmul(VxH) to convert to embedding
# Followed by an XentLoss layer which needs to handle sparse dy, dense y
#
# TODO:
# - make sure Mmul and XentLoss can handle sparse input
# - Find a way to recover hidden state after back for truncated BPTT.
# - registers: out, dif, dif-incr
# - we want to recover out, the hidden states are there
# - we can get rid of dif and dif-incr? yes we have already performed the update.
# - actually the states that we want are already in out0! we can recover them after back.
# - train should pass a keepstate option down to backprop->back and back should recover the state.
# - or the next forw should set out=out0 instead of out=nothing.
# - initsequence should do that instead of out=nothing.
