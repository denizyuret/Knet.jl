VERSION < v"0.6-" && error("currenctly modular interface only works for julia v0.6+")

using Knet

# load data

const data = readstring(Knet.dir("data","10.txt"));
const dict = unique(data);

init(x...) = .2rand(x...) .- .1

# define model

const model = let
    encoder = Embedding(length(dict), 64, init=init)
    lstm1   = LSTM(64, 256, init=init)
    lstm2   = LSTM(256, 256, init=init)
    decoder = Affine(256, length(dict), init=init)

    function CharLM(x, h)
        h1, c1, h2, c2 = h
        
        input  = encoder(x)
        h1, c1 = lstm1(input, h1, c1)
        h2, c2 = lstm2(h1,    h2, c2)
        result = logp(decoder(h2), 2)

        result, (h1, c1, h2, c2)
    end
end

# train model

function bptt!(seq, seqlen=length(seq), batchsize=length(seq[1]))
    seq = track(seq) # enable auto diff
    h   = ntuple(i->zeros(batchsize, 256), 4)

    loss = 0
    for i in 1:seqlen-1
        y = getval(seq)[i+1]
        pred, h = model(seq[i], h)
        index = map(1:batchsize) do i
            batchsize * (y[i] - 1) + i
        end
        loss += -sum(pred[index])
    end

    println("loss: ", getval(loss))

    back!(loss, 1)
end

for epoch in 1:1000
    println("epoch: $epoch")

    seqlen = min(20 + epoch, 150+rand(0:5))
    batchsize = 32

    for nbatch in 0:length(data)÷(batchsize*seqlen)-2
        seq = map(1:seqlen) do i
            [findfirst(dict, data[nbatch*batchsize*seqlen + j*seqlen + i]) for j in 0:batchsize-1]
        end

        bptt!(seq)

        for p in params(model)
            update!(getval(p), getgrad(p), lr=.001)
        end
    end
end

# generate a seq

function sample(p)
    r = rand()
    for c = 1:length(p)
        r -= p[c]
        r <= 0 && return dict[c]
    end
end

h, last = ntuple(i->zeros(1, 256), 4), '\n'

for i in 1:800
    pred, h = model([findfirst(dict, last)], h)
    last = sample(exp.(pred))
    print(last)
end
