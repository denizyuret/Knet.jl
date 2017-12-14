for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""

julia bilstm-tagger-withchar.jl # to use with default options on CPU
julia bilstm-tagger-withchar.jl --usegpu # to use with default options on GPU
julia bilstm-tagger-withchar.jl -h # to see all options with default values

This example implements a named entity tagger built on top of a BiLSTM
neural network similar to the model defined in 'Bidirectional LSTM-CRF Models
for Sequence Tagging', Zhiheng Huang, Wei Xu, Kai Yu, arXiv technical report
1508.01991, 2015. Originally, this model implemented for dynet-benchmarks.
This model generates UNK words' embeddings by using an another BiLSTM network
which takes characters as input.

* Paper url: https://arxiv.org/pdf/1508.01991.pdf
* DyNet report: https://arxiv.org/abs/1701.03980
* Benchmark repo: https://github.com/neulab/dynet-benchmark

"""
module CharTagger
using Knet
using ArgParse

include(Pkg.dir("Knet","data","wikiner.jl"))
const F = Float32
t00 = now()

function main(args)
    s = ArgParseSettings()
    s.description = "Bidirectional LSTM Tagger (with chars) in Knet"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--cembed"; arg_type=Int; default=20; help="char embedding size")
        ("--wembed"; arg_type=Int; default=128; help="word embedding size")
        ("--hidden"; arg_type=Int; default=50; help="BiLSTM hidden size")
        ("--mlp"; arg_type=Int; default=32; help="MLP size")
        ("--timeout"; arg_type=Int; help="max timeout (in seconds)"; default=600)
        ("--epochs"; arg_type=Int; default=100; help="epochs")
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
    w, srnns = initweights(
        atype, o[:hidden], length(data.w2i), data.ntags, data.nchars,
        o[:wembed], o[:cembed], o[:mlp], o[:usegpu])
    opt = optimizers(w, Adam)

    # train bilstm tagger
    nwords = data.nwords; ntags = data.ntags; nchars = data.nchars
    println("nwords=$nwords, ntags=$ntags, nchars=$nchars"); flush(STDOUT)
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
                    input = make_input(sent, data.w2i, data.c2i)
                    nwords = length(sent)
                    ypred,_ = predict(w, input, srnns)
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

            # train with instance
            input = make_input(data.trn[k],data.w2i,data.c2i)
            out = make_output(data.trn[k],data.t2i)
            batch_loss = train!(w,input,out,srnns,opt)
            this_loss += batch_loss
            this_tagged += length(data.trn[k])
        end
        @printf("epoch %d finished\n", epoch-1); flush(STDOUT)
    end
end

function make_input(sample, w2i, c2i)
    seq, is_word = Any[], Bool[]
    words = map(x->x[1], sample)
    for word in words
        push!(is_word, haskey(w2i, word) && word != UNK)
        if is_word[end]
            push!(seq, w2i[word])
        else
            chars = [PAD; convert(Array{UInt8,1}, word); PAD]
            push!(seq, convert(Array{Int32}, map(c->c2i[c], chars)))
        end
    end

    # construct rare word block - it's for efficiency
    rare_words = filter(x->isa(x,Array), seq)
    sort!(rare_words, by=x->length(x), rev=true)
    longest = length(rare_words) != 0 ? length(rare_words[1]) : 0
    rblock = Int32[]
    batchsizes = zeros(Int32, longest)
    for t = 1:longest
        for i = 1:length(rare_words)
            length(rare_words[i]) < t && break
            push!(rblock, rare_words[i][t])
            batchsizes[t] += 1
        end
    end

    cinds = find(is_word)
    rinds = find(.!is_word)
    cwords = seq[is_word]
    cwords = reshape(cwords, 1, length(cwords))
    cwords = convert(Array{Int32}, cwords)
    rwords = rblock

    return (cwords, cinds, rwords, rinds, vec(batchsizes))
end

function make_output(sample,t2i)
    map(s->t2i[s[2]], sample)
end

function initweights(
    atype, hidden, words, tags, chars, wembed, cembed, mlp, usegpu, winit=0.01)
    w = Array{Any}(8)
    _birnninit(x,y) = rnninit(x,y; bidirectional=true, usegpu=usegpu)

    # init rnns
    srnn1, wrnn1 = _birnninit(wembed, hidden)
    w[1] = wrnn1
    srnn2, wrnn2 = _birnninit(cembed, div(wembed,2))
    w[2] = wrnn2

    # weight/bias params for MLP network
    w[3] = convert(atype, winit*randn(mlp, 2*hidden))
    w[4] = convert(atype, zeros(mlp, 1))
    w[5] = convert(atype, winit*randn(tags, mlp))
    w[6] = convert(atype, winit*randn(tags, 1))

    # word/char embeddings
    w[7] = convert(atype, winit*randn(wembed, words))
    w[8] = convert(atype, winit*randn(cembed, chars))
    return w, [srnn1, srnn2]
end

# loss function
function loss(w, input, ygold, srnns)
    py, _ = predict(w,input,srnns)
    return nll(py,ygold)
end

lossgradient = gradloss(loss)

function predict(w,input,srnns)
    x = encoder(w,input,srnns[2])
    x = reshape(x, size(x,1), 1, size(x,2))
    r = srnns[1]; wr = w[1]
    wmlp, bmlp = w[3], w[4]
    wy, by = w[5], w[6]
    y, hy, cy = rnnforw(r,wr,x)
    y2 = reshape(y,size(y,1),size(y,2)*size(y,3))
    y3 = wmlp * y2 .+ bmlp
    return wy*y3.+by, hy, cy
end

# encoder - it generates embeddings
function encoder(w,input,srnn)
    # expand input tuple
    cwords, cinds, rwords, rinds, bs = input

    # common words' embedding
    cembed = w[end-1][:,cwords]
    cembed = reshape(cembed, size(cembed,1), size(cembed)[end])
    length(rinds) == 0 && return cembed

    r = srnn; wr = w[2]
    c0 = w[end][:,rwords]
    y, hy, cy = rnnforw(r,wr,c0; hy=true, cy=true, batchSizes=bs)
    r0 = permutedims(hy, (3,1,2))
    rembed = reshape(r0, size(r0,1)*size(r0,2), size(r0,3))

    e0 = hcat(cembed, rembed)
    e1 = e0[:,[cinds...,rinds...]]
    return e1
end

function train!(w,input,ygold,srnns,opt)
    gloss, lossval = lossgradient(w, input, ygold, srnns)
    for k = 1:length(w)
        gloss[k] != nothing && update!(w[k], gloss[k], opt[k])
    end
    return lossval*(length(input[2])+length(input[4]))
end

splitdir(PROGRAM_FILE)[end] == "bilstm-tagger-withchar.jl" && main(ARGS)
end # module
