module Tagger
using Knet
using AutoGrad
using ArgParse

const train_file = "data/tags/train.txt"
const test_file = "data/tags/dev.txt"
const UNK = "_UNK_"
t00 = now()

function main(args)
    s = ArgParseSettings()
    s.description = "Bidirectional LSTM Tagger in Knet"

    @add_arg_table s begin
        ("--gpu"; action=:store_true; help="use GPU or not")
        ("EMBED_SIZE"; arg_type=Int; help="embedding size")
        ("HIDDEN_SIZE"; arg_type=Int; help="hidden size")
        ("MLP_SIZE"; arg_type=Int; help="MLP size")
        ("SPARSE"; arg_type=Int; help="sparse update 0/1")
        ("TIMEOUT"; arg_type=Int; help="max timeout")
        ("--batchsize"; arg_type=Int; help="minibatch size"; default=1)
        ("--train"; default=train_file; help="train file")
        ("--test"; default=test_file; help="test file")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--epochs"; arg_type=Int; default=100; help="epochs")
        ("--minoccur"; arg_type=Int; default=6)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.setseed(o[:seed])
    atype = o[:gpu] ? KnetArray{Float32} : Array{Float32}

    # read data
    trn = read_file(o[:train])
    tst = read_file(o[:test])

    # get words and tags from train set
    words, tags = [], []
    for sample in trn
        for (word,tag) in sample
            push!(words, word)
            push!(tags, tag)
        end
    end

    # count words and build vocabulary
    wordcounts = count_words(words)
    nwords = length(wordcounts)+1
    wordcounts = filter((w,c)-> c >= o[:minoccur], wordcounts)
    words = collect(keys(wordcounts))
    !in(UNK, words) && push!(words, UNK)
    w2i, i2w = build_vocabulary(words)
    t2i, i2t = build_vocabulary(tags)
    ntags = length(t2i)
    !haskey(w2i, UNK) && error("...")

    # build model
    w, srnn = initweights(atype, o[:HIDDEN_SIZE], length(w2i), length(t2i),
                    o[:MLP_SIZE], o[:EMBED_SIZE])
    opt = optimizers(w, Adam)

    # train bilstm tagger
    println("nwords=$nwords, ntags=$ntags"); flush(STDOUT)
    println("startup time: ", Int(now()-t00)*0.001); flush(STDOUT)
    t0 = now()
    all_time = dev_time = all_tagged = this_tagged = this_loss = 0
    for epoch = 1:o[:epochs]
        shuffle!(trn)
        for k = 1:length(trn)
            iter = (epoch-1)*length(trn) + k
            if iter % 500 == 0
                @printf("%f\n", this_loss/this_tagged); flush(STDOUT)
                all_tagged += this_tagged
                this_loss = this_tagged = 0
                all_time = Int(now()-t0)*0.001
            end

            if iter % 10000 == 0 || all_time > o[:TIMEOUT]
                dev_start = now()
                good_sent = bad_sent = good = bad = 0.0
                for sent in tst
                    seq = make_input(sent, w2i)
                    nwords = length(sent)
                    ypred,_ = predict(w, seq, srnn)
                    ypred = map(x->i2t[x], mapslices(indmax,Array(ypred),1))
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
                dev_time += Int(now()-dev_start)*0.001
                train_time = Int(now()-t0)*0.001-dev_time

                @printf(
                    "tag_acc=%.4f, sent_acc=%.4f, time=%.4f, word_per_sec=%.4f\n",
                    good/(good+bad), good_sent/(good_sent+bad_sent), train_time,
                    all_tagged/train_time); flush(STDOUT)

                all_time > o[:TIMEOUT] && return
            end

            # train on minibatch
            x = make_input(trn[k], w2i)
            y = make_output(trn[k], t2i)

            batch_loss = train!(w,x,y,srnn,opt)
            this_loss += batch_loss
            this_tagged += length(trn[k])
        end
        @printf("epoch %d finished\n", epoch-1); flush(STDOUT)
    end
end

# parse line
function parse_line(line)
    return map(x->split(x,"|"), split(replace(line,"\n",""), " "))
end

# read file
function read_file(file)
    data = open(file, "r") do f
        map(parse_line, readlines(f))
    end
end

function count_words(words)
    wordcounts = Dict()
    for word in words
        wordcounts[word] = get(wordcounts, word, 0) + 1
    end
    return wordcounts
end

function build_vocabulary(words)
    words = collect(Set(words))
    w2i = Dict(); i2w = Dict()
    counter = 1
    for (i,word) in enumerate(words)
        w2i[word] = i
        i2w[i] = word
    end
    w2i, i2w
end

# make input
function make_input(sample, w2i)
    nwords = length(sample)
    x = map(i->get(w2i, sample[i][1], w2i[UNK]), [1:nwords...])
    x = reshape(x,1,length(x))
    x = convert(Array{Int64}, x)
end

# make output
function make_output(sample, t2i)
    nwords = length(sample)
    y = map(i->t2i[sample[i][2]], [1:nwords...])
    y = reshape(y,1,length(y))
    y = convert(Array{Int64}, y)
end

# initialize all weights of the language model
# w[1:2] => weight/bias params for forward LSTM network
# w[3:4] => weight/bias params for backward LSTM network
# w[5:8] => weight/bias params for MLP network
# w[9]   => word embeddings
function initweights(atype, hidden, words, tags, embed, mlp, winit=0.01)
    w = Array(Any, 6)
    input = embed
    srnn, wrnn = rnninit(input, hidden; bidirectional=true)
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

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE=="knet/bilstm-tagger.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end # module
