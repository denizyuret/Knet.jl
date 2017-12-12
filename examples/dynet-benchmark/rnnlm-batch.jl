module RNNLM
using Knet
using AutoGrad
using ArgParse

const train_file = "data/text/train.txt"
const test_file = "data/text/dev.txt"
const SOS = "<s>"
t00 = now()

function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "RNN Language Model in Knet"
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
        ("--gpu"; action=:store_true; help="use GPU or not")
        ("MB_SIZE"; arg_type=Int; help="minibatch_size")
        ("EMBED_SIZE"; arg_type=Int; help="embedding size")
        ("HIDDEN_SIZE"; arg_type=Int; help="hidden size")
        ("SPARSE"; arg_type=Int; help="sparse update 0/1")
        ("TIMEOUT"; arg_type=Int; help="max timeout")
        ("--train"; default=train_file; help="train file")
        ("--test"; default=test_file; help="test file")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--epochs"; arg_type=Int; default=100; help="epochs")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.setseed(o[:seed])
    atype = o[:gpu] ? KnetArray{Float32} : Array{Float32}

    # build data
    w2i = Dict()
    trn = read_data(o[:train], w2i)
    tst = read_data(o[:test], w2i)
    sort!(trn, by=length, rev=true)
    sort!(tst, by=length, rev=true)
    trn, tst = map(split->make_batches(split, w2i, o[:MB_SIZE]), [trn, tst])

    # build model
    w,srnn = initweights(atype, o[:HIDDEN_SIZE], length(w2i), o[:EMBED_SIZE])
    opt = optimizers(w, Adam)

    # train language model
    println("startup time: ", Int(now()-t00)*0.001); flush(STDOUT)
    t0 = now()
    all_time = dev_time = all_tagged = this_words = this_loss = 0
    for epoch = 1:o[:epochs]
        shuffle!(trn)
        for k = 1:length(trn)
            iter = (epoch-1)*length(trn) + k
            if iter % div(500, o[:MB_SIZE]) == 0
                @printf("%f\n", this_loss/this_words); flush(STDOUT)
                all_tagged += this_words
                this_loss = this_words = 0
                all_time = Int(now()-t0)*0.001
            end

            if iter % div(10000, o[:MB_SIZE]) == 0
                dev_start = now()
                dev_loss = dev_words = 0
                for i = 1:length(tst)
                    x, y, nwords = tst[i]
                    dev_loss += loss(w,x,y,srnn)*nwords
                    dev_words += nwords
                end
                dev_time += Int(now()-dev_start)*0.001
                train_time = Int(now()-t0)*0.001-dev_time

                @printf(
                    "nll=%.4f, ppl=%.4f, words=%d, time=%.4f, word_per_sec=%.4f\n",
                    dev_loss/dev_words, exp(dev_loss/dev_words), dev_words,
                    train_time, all_tagged/train_time); flush(STDOUT)

                if all_time > o[:TIMEOUT]
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
    get_tokens(line) = [split(line, " ")[2:end-1]; SOS]
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
end

# make minibatches
function make_batches(data, w2i, batchsize)
    batches = []
    for k = 1:batchsize:length(data)
        samples = data[k:min(k+batchsize-1, length(data))]
        lengths = map(length, samples)
        longest = reduce(max, lengths)
        nwords = sum(lengths)
        nsamples = length(samples)
        pad = length(w2i)
        seq = pad*ones(nsamples,longest+1)
        for i = 1:nsamples
            map!(t->seq[i,t] = samples[i][t], [1:length(samples[i])...])
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
function initweights(atype, hidden, vocab, embed, winit=0.01)
    w = Array(Any, 4)
    input = embed

    # rnn
    # w[1] = winit*randn(4*hidden, hidden+input)
    # w[2] = zeros(4*hidden, 1)
    # w[2][1:hidden] = 1 # forget gate bias
    srnn,wrnn = rnninit(input,hidden)
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

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE=="knet/rnnlm-batch.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end # module
