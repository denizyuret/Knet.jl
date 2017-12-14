for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""

julia rnnlm-batch.jl # to use with default options on CPU
julia rnnlm-batch.jl --usegpu # to use with default options on GPU
julia rnnlm-batch.jl -h # to see all options with default values

This example implements a standard RNN language model on top of LSTM cells.
This example is originally implemented for dynet-benchmark repo.

* Benchmark repo: https://github.com/neulab/dynet-benchmark

"""
module RNNLM
using Knet
using ArgParse

const SOS = "<s>"
include(Pkg.dir("Knet","data","mikolovptb.jl"))
t00 = now()

function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "RNN Language Model in Knet"
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--batchsize"; arg_type=Int; help="minibatch_size"; default=64)
        ("--embed"; arg_type=Int; help="word embedding size"; default=256)
        ("--hidden"; arg_type=Int; help="lstm hidden size"; default=128)
        ("--sparse"; arg_type=Int; help="sparse update 0/1"; default=0)
        ("--timeout"; arg_type=Int; default=600; help="max timeout (in seconds)")
        ("--epochs"; arg_type=Int; default=100; help="number of training epochs")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.setseed(o[:seed])
    atype = o[:usegpu] ? KnetArray{Float32} : Array{Float32}
    datadir = abspath(joinpath(@__DIR__, "../data/text"))

    trn = dev = nothing; vocabsize = 0;
    if isdir(datadir)
        w2i = Dict()
        trnfile = abspath(joinpath(datadir, "train.txt"))
        devfile = abspath(joinpath(datadir, "dev.txt"))
        trn = read_data(trnfile, w2i)
        dev = read_data(devfile, w2i)
        vocabsize = length(w2i)+1
    else
        trn, dev, _, i2w = mikolovptb()
        vocabsize = length(i2w)+1
        sort!(trn, by=length, rev=true)
        sort!(dev, by=length, rev=true)
    end
    trn, dev = map(s->make_batches(s, vocabsize, o[:batchsize]), [trn, dev])

    # build model
    w,srnn = initweights(atype, o[:hidden], vocabsize, o[:embed], o[:usegpu])
    opt = optimizers(w, Adam)

    # train language model
    println("startup time: ", Int((now()-t00).value)*0.001); flush(STDOUT)
    t0 = now()
    all_time = dev_time = all_tagged = this_words = this_loss = 0
    o[:timeout] = o[:timeout] <= 0 ? Inf : o[:timeout]
    for epoch = 1:o[:epochs]
        shuffle!(trn)
        for k = 1:length(trn)
            iter = (epoch-1)*length(trn) + k
            if iter % div(500, o[:batchsize]) == 0
                @printf("%f\n", this_loss/this_words); flush(STDOUT)
                all_tagged += this_words
                this_loss = this_words = 0
                all_time = Int((now()-t0).value)*0.001
            end

            if iter % div(10000, o[:batchsize]) == 0
                dev_start = now()
                dev_loss = dev_words = 0
                for i = 1:length(dev)
                    x, y, nwords = dev[i]
                    dev_loss += loss(w,x,y,srnn)*nwords
                    dev_words += nwords
                end
                dev_time += Int((now()-dev_start).value)*0.001
                train_time = Int((now()-t0).value)*0.001-dev_time

                @printf(
                    "nll=%.4f, ppl=%.4f, words=%d, time=%.4f, word_per_sec=%.4f\n",
                    dev_loss/dev_words, exp(dev_loss/dev_words), dev_words,
                    train_time, all_tagged/train_time); flush(STDOUT)

                if all_time > o[:timeout]
                    return
                end
            end

            # train on minibatch
            x, y, batch_words = trn[k]
            batch_loss = train!(w,x,y,opt,srnn)
            this_loss += batch_loss*batch_words
            this_words += batch_words
        end
        @printf("epoch %d finished\n", epoch-1); flush(STDOUT)
    end
end

# build vocabulary, training and test data
function read_data(file, w2i)
    get_tokens(line) = split(line, " ")[2:end-1]
    data = open(file, "r") do f
        data = []
        for ln in readlines(f)
            words = get_tokens(ln)
            senvec = []
            for word in words
                if !haskey(w2i, word)
                    w2i[word] = length(w2i)+1
                end
                push!(senvec, w2i[word])
            end
            push!(data, senvec)
        end
        data
    end
    sort!(data, by=length, rev=true)
end

# make minibatches
function make_batches(data, vocabsize, batchsize)
    batches = []
    for k = 1:batchsize:length(data)
        samples = data[k:min(k+batchsize-1, length(data))]
        lengths = map(length, samples)
        longest = reduce(max, lengths)
        nwords = sum(lengths)
        nsamples = length(samples)
        pad = vocabsize
        seq = pad*ones(nsamples,longest+1)
        for i = 1:nsamples
            for t = 1:length(samples[i])
                seq[i,t] = samples[i][t]
            end
        end
        x = seq[:,1:end-1]
        x = convert(Array{Int64}, x)
        y = seq[:,2:end]
        y = convert(Array{Int64}, y)
        push!(batches, (x, y, nwords))
    end
    return batches
end

# initialize all weights of the language model
# w[1:2] => weight/bias params for LSTM network
# w[3:4] => weight/bias params for softmax layer
# w[5]   => word embeddings
function initweights(atype, hidden, vocab, embed, usegpu, winit=0.01)
    w = Array{Any}(4)
    input = embed

    # rnn
    # w[1] = winit*randn(4*hidden, hidden+input)
    # w[2] = zeros(4*hidden, 1)
    # w[2][1:hidden] = 1 # forget gate bias
    srnn,wrnn = rnninit(input,hidden; usegpu=usegpu)
    w[1] = wrnn

    # softmax
    w[2] = convert(atype, winit*randn(vocab+1, hidden))
    w[3] = convert(atype, zeros(vocab+1, 1))

    # embed
    w[4] = convert(atype, winit*randn(embed, vocab+1))
    return w, srnn
end

function predict(ws,xs,srnn,hx=nothing,cx=nothing)
    wx = ws[4]; r = srnn; wr = ws[1]; wy = ws[2]; by = ws[3]
    x = wx[:,xs]
    y, hy, cy = rnnforw(r,wr,x,hx,cx)
    y2 = reshape(y,size(y,1),size(y,2)*size(y,3))
    return wy*y2.+by, hy, cy
end

function loss(w,x,y,srnn,h=nothing,c=nothing)
    py,hy,cy = predict(w,x,srnn,h,c)
    return nll(py,y; average=true)
end

lossgradient = gradloss(loss)

function train!(w,x,y,opt,srnn,h=nothing,c=nothing)
    gloss,lossval = lossgradient(w,x,y,srnn,h,c)
    update!(w, gloss, opt)
    return lossval
end

splitdir(PROGRAM_FILE)[end] == "rnnlm-batch.jl" && main(ARGS)
end # module
