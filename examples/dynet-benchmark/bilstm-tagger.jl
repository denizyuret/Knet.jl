for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""

julia bilstm-tagger.jl # to use with default options on CPU
julia bilstm-tagger.jl --usegpu # to use with default options on GPU
julia bilstm-tagger.jl -h # to see all options with default values

This example implements a named entity tagger built on top of a BiLSTM
neural network similar to the model defined in 'Bidirectional LSTM-CRF Models
for Sequence Tagging', Zhiheng Huang, Wei Xu, Kai Yu, arXiv technical report
1508.01991, 2015. Originally, this model implemented for dynet-benchmarks.

* Paper url: https://arxiv.org/pdf/1508.01991.pdf
* DyNet report: https://arxiv.org/abs/1701.03980
* Benchmark repo: https://github.com/neulab/dynet-benchmark

"""
module Tagger
using Knet
using ArgParse

include(Pkg.dir("Knet","data","wikiner.jl"))
t00 = now()

function main(args)
    s = ArgParseSettings()
    s.description = "Bidirectional LSTM Tagger in Knet"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--embed"; arg_type=Int; default=128; help="word embedding size")
        ("--hidden"; arg_type=Int; default=50; help="LSTM hidden size")
        ("--mlp"; arg_type=Int; default=32; help="MLP size")
        ("--timeout"; arg_type=Int; default=600; help="max timeout (in seconds)")
        ("--epochs"; arg_type=Int; default=100; help="number of training epochs")
        ("--minoccur"; arg_type=Int; default=6; help="word min occurence limit")
        ("--report"; arg_type=Int; default=500; help="report period in iters")
        ("--valid"; arg_type=Int; default=10000; help="valid period in iters")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.setseed(o[:seed])
    atype = o[:atype] = !o[:usegpu] ? Array{Float32} : KnetArray{Float32}
    datadir = abspath(joinpath(@__DIR__, "../data/tags"))
    datadir = isdir(datadir) ? datadir : WIKINER_DIR

    # load WikiNER data
    data = WikiNERData(datadir, o[:minoccur])

    # build model
    nwords = length(data.w2i); ntags = data.ntags
    w, srnn = initweights(
        atype, o[:hidden], nwords, ntags, o[:mlp], o[:embed], o[:usegpu])
    opt = optimizers(w, Adam)

    # train bilstm tagger
    nwords = data.nwords; ntags = data.ntags
    println("nwords=$nwords, ntags=$ntags"); flush(STDOUT)
    println("startup time: ", Int((now()-t00).value)*0.001); flush(STDOUT)
    t0 = now()
    all_time = dev_time = all_tagged = this_tagged = this_loss = 0
    o[:timeout] = o[:timeout] <= 0 ? Inf : o[:timeout]
    for epoch = 1:o[:epochs]
        shuffle!(data.trn)
        for k = 1:length(data.trn)
            iter = (epoch-1)*length(data.trn) + k
            if o[:report] > 0 && iter % o[:report] == 0
                @printf("%f\n", this_loss/this_tagged); flush(STDOUT)
                all_tagged += this_tagged
                this_loss = this_tagged = 0
                all_time = Int((now()-t0).value)*0.001
            end

            if all_time > o[:timeout] || o[:valid] > 0 && iter % o[:valid] == 0
                dev_start = now()
                good_sent = bad_sent = good = bad = 0.0
                for sent in data.dev
                    seq = make_input(sent, data.w2i)
                    nwords = length(sent)
                    ypred,_ = predict(w, seq, srnn)
                    ypred = map(
                        x->data.i2t[x], mapslices(indmax,Array(ypred),1))
                    ygold = map(x -> x[2], sent)
                    same = true
                    for (y1,y2) in zip(ypred, ygold)
                        if y1 == y2
                            good += 1
                        else
                            bad += 1
                            same = false
                        end
                    end
                    if same
                        good_sent += 1
                    else
                        bad_sent += 1
                    end
                end
                dev_time += Int((now()-dev_start).value)*0.001
                train_time = Int((now()-t0).value)*0.001-dev_time

                @printf(
                    "tag_acc=%.4f, sent_acc=%.4f, time=%.4f, word_per_sec=%.4f\n",
                    good/(good+bad), good_sent/(good_sent+bad_sent), train_time,
                    all_tagged/train_time); flush(STDOUT)

                all_time > o[:timeout] && return
            end

            # train on minibatch
            x = make_input(data.trn[k], data.w2i)
            y = make_output(data.trn[k], data.t2i)
            batch_loss = train!(w,x,y,srnn,opt)
            this_loss += batch_loss
            this_tagged += length(data.trn[k])
        end
        @printf("epoch %d finished\n", epoch-1); flush(STDOUT)
    end
end

function make_input(sample, w2i)
    nwords = length(sample)
    x = map(i->get(w2i, sample[i][1], w2i[UNK]), [1:nwords...])
    x = reshape(x,1,length(x))
    x = convert(Array{Int64}, x)
end

function make_output(sample, t2i)
    nwords = length(sample)
    y = map(i->t2i[sample[i][2]], [1:nwords...])
    y = reshape(y,1,length(y))
    y = convert(Array{Int64}, y)
end

# w[1]   => weight/bias params for forward LSTM network
# w[2:5] => weight/bias params for MLP+softmax network
# w[6]   => word embeddings
function initweights(atype, hidden, words, tags, embed, mlp, usegpu, winit=0.01)
    w = Array{Any}(6)
    input = embed
    srnn, wrnn = rnninit(input, hidden; bidirectional=true, usegpu=usegpu)
    w[1] = wrnn
    w[2] = convert(atype, winit*randn(mlp, 2*hidden))
    w[3] = convert(atype, zeros(mlp, 1))
    w[4] = convert(atype, winit*randn(tags, mlp))
    w[5] = convert(atype, winit*randn(tags, 1))
    w[6] = convert(atype, winit*randn(embed, words))
    return w, srnn
end

# loss function
function loss(w, x, ygold, srnn, h=nothing, c=nothing)
    py, _ = predict(w,x,srnn,h,c)
    return nll(py,ygold)
end

function predict(ws,xs,srnn,hx=nothing,cx=nothing)
    wx = ws[6]
    r = srnn; wr = ws[1]
    wmlp = ws[2]; bmlp = ws[3];
    wy = ws[4]; by = ws[5]
    x = wx[:,xs]
    y, hy, cy = rnnforw(r,wr,x)
    y2 = reshape(y,size(y,1),size(y,2)*size(y,3))
    y3 = wmlp * y2 .+ bmlp
    return wy*y3.+by, hy, cy
end

lossgradient = gradloss(loss)

function train!(w,x,y,srnn,opt,h=nothing,c=nothing)
    gloss, lossval = lossgradient(w, x, y, srnn)
    update!(w,gloss,opt)
    return lossval*size(x,2)
end

splitdir(PROGRAM_FILE)[end] == "bilstm-tagger.jl" && main(ARGS)
end # module
