# Parameters from the Zaremba implementation:
batch_size=20
seq_length=20
layers=2
decay=2
rnn_size=200
dropout=0
init_weight=0.1
lr=1
vocab_size=10000
max_epoch=4
max_max_epoch=13
max_grad_norm=5

import Base: start, next, done

type LMData; data; dict; batchsize; seqlength; batch; end

function LMData(fname::String; batch=batch_size, seqlen=seq_length, dict=Dict{Any,Int32}())
    data = Int32[]
    f = open(fname)
    for l in eachline(f)
        for w in split(l)
            push!(data, get!(dict, w, 1+length(dict)))
        end
        push!(data, get!(dict, "<eos>", 1+length(dict))) # end-of-sentence
    end
    x = [ speye(Float64, length(dict), batch) for i=1:seqlen+1 ]
    info("Read $fname: $(length(data)) words, $(length(dict)) vocab.")
    LMData(data, dict, batch, seqlen, x)
end

function start(d::LMData)
    mx = size(d.batch[1], 1)
    nd = length(d.dict)
    if nd > mx                  # if dict size increases, adjust batch arrays
        for x in d.batch; x.m = nd; end
    elseif nd  < mx
        error("Dictionary shrinkage")
    end
    return 0
end

function done(d::LMData,nword)
    # nword is the number of sequences served
    # stop if there is not enough data for another batch
    nword + d.batchsize * d.seqlength > length(d.data)
end

function next(d::LMData,nword)                              # d.data is the whole corpus represented as a sequence of Int32's
    segsize = div(length(d.data), d.batchsize)          # we split it into d.batchsize roughly equal sized segments
    offset = div(nword, d.batchsize) # this is how many words have been served so far from each segment
    for b = 1:d.batchsize
        idata = (b-1)*segsize + offset                  # start producing words in segment b at x=data[idata], y=data[idata+1]
        for t = 1:d.seqlength+1
            d.batch[t].rowval[b] = d.data[idata+t]
        end
    end
    xbatch = d.batch[1:d.seqlength]
    ybatch = d.batch[2:d.seqlength+1]
    ((xbatch, ybatch), nword + d.seqlength * d.batchsize)	# each call to next will deliver d.seqlength words from each of the d.batchsize segments
end


dir = "data"
trn = LMData("$dir/ptb.train.txt")
dev = LMData("$dir/ptb.valid.txt"; dict=trn.dict)
tst = LMData("$dir/ptb.test.txt"; dict=trn.dict)

vocab_size = length(trn.dict)   #DBG
@assert length(trn.dict) == vocab_size "$((length(trn.dict), vocab_size))"

using KUnet

lstms = Any[]
for i=1:layers; push!(lstms, LSTM(rnn_size; dropout=dropout)); end
net = Net(Mmul(rnn_size), lstms..., Mmul(vocab_size), Bias(), Soft(), SoftLoss())
setparam!(net, lr=lr, init=rand!, initp=(-init_weight, init_weight))
setseed(42)

for ep=1:max_max_epoch
    ep > max_epoch && (lr /= decay; setparam!(net, lr=lr))
    @time (ltrn,w,g) = train(net, trn; gclip=max_grad_norm, keepstate=true)
    ptrn = exp(ltrn/trn.seqlength)
    @time ldev = test(net, dev; keepstate=true)
    pdev = exp(ldev/dev.seqlength)
    @time ltst = test(net, tst; keepstate=true)
    ptst = exp(ltst/tst.seqlength)
    @show (ep, ptrn, pdev, ptst, w, g)
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
