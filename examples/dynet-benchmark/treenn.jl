for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""

julia treenn.jl # to use with default options on CPU
julia treenn.jl --usegpu # to use with default options on GPU
julia treenn.jl -h # to see all options with default values

This example implements a binary tree-structured LSTM networks proposed
in 'Improved Semantic Representations From Tree-Structured Long Short-Term
Memory Networks', Kai Sheng Tai, Richard Socher, Christopher D. Manning,
arXiv technical report 1503.00075, 2015.

* Paper url: https://arxiv.org/pdf/1503.00075.pdf
* Project page: https://github.com/stanfordnlp/treelstm

"""
module TreeLSTM
using Knet
using ArgParse

const UNK = "_UNK_"
t00 = now()
include(Pkg.dir("Knet","data","treebank.jl"))

function main(args)
    s = ArgParseSettings()
    s.description = "Tree-structured LSTM network in Knet."

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--embed"; arg_type=Int; default=128; help="word embedding size")
        ("--hidden"; arg_type=Int; default=128; help="LSTM hidden size")
        ("--timeout"; arg_type=Int; default=600; help="max timeout (in seconds)")
        ("--epochs"; arg_type=Int; default=3; help="number of training epochs")
        ("--minoccur"; arg_type=Int; default=0; help="word min occurence limit")
        ("--report"; arg_type=Int; default=1000; help="report period (in iters)")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.setseed(o[:seed])
    atype = o[:atype] = !o[:usegpu] ? Array{Float32} : KnetArray{Float32}
    datadir = abspath(joinpath(@__DIR__, "../data/trees"))
    datadir = isdir(datadir) ? datadir : TREEBANK_DIR

    # read data
    trn, dev = load_treebank_data(datadir)

    # build vocabs
    l2i, w2i, i2l, i2w = build_treebank_vocabs(trn)
    nwords = length(w2i); nlabels = length(l2i)

    make_data!(trn, w2i, l2i)
    make_data!(dev, w2i, l2i)

    # build model
    w = initweights(atype, o[:hidden], nwords, nlabels, o[:embed])
    opt = optimizers(w, Adam)

    # main loop
    println("startup time: ", Int((now()-t00).value)*0.001); flush(STDOUT)
    all_time = sents = 0
    o[:timeout] = o[:timeout] <= 0 ? Inf : o[:timeout]
    for epoch = 1:o[:epochs]
        closs = cwords = 0
        shuffle!(trn)
        t0 = now()
        for k = 1:length(trn)
            sents += 1
            iter = (epoch-1)*length(trn) + k
            tree = trn[k]
            this_loss, this_words = train!(w,tree,opt)
            closs += this_loss
            cwords += this_words

            if o[:report] > 0 && iter % o[:report] == 0
                @printf("%f\n", closs/cwords); flush(STDOUT)
                closs = cwords = 0
            end
        end
        all_time += Int((now()-t0).value)*0.001

        good = bad = 0
        for tree in dev
            ind, nwords = predict(w, tree)
            ypred = i2l[ind]
            ygold = tree.label
            if ypred == ygold
                good += 1
            else
                bad += 1
            end
        end
        @printf(
            "acc=%.4f, time=%.4f, sent_per_sec=%.4f\n",
            good/(good+bad), all_time, sents/all_time); flush(STDOUT)

        all_time > o[:timeout] && return
    end
end

function initweights(atype, hidden, words, labels, embed, winit=0.01)
    w = Array{Any}(9)
    w[1] = winit*randn(3*hidden, embed)
    w[2] = zeros(3*hidden, 1)
    w[3] = winit*randn(3*hidden, 2*hidden)
    w[4] = zeros(3*hidden, 1)
    w[5] = winit*randn(hidden, hidden)
    w[6] = winit*randn(hidden, hidden)
    w[7] = ones(hidden,1)
    w[8] = winit*randn(labels, hidden)
    w[9] = winit*randn(embed, words)
    return map(i->convert(atype, i), w)
end

function lstm(w,ind)
    x = w[end][:,ind]
    x = reshape(x, length(x), 1)
    hsize = size(x,1)
    gates = w[1] * x .+ w[2]
    i = sigm.(gates[1:hsize,:])
    o = sigm.(gates[1+hsize:2hsize,:])
    u = sigm.(gates[1+2hsize:3hsize,:])
    c = i .* u
    h = o .* tanh.(c)
    return (h,c)
end

function slstm(w,h1,h2,c1,c2)
    hsize = size(h1,1)
    h = vcat(h1,h2)
    gates = w[3] * h .+ w[4]
    i  = sigm.(gates[1:hsize,:])
    o  = sigm.(gates[1+hsize:2hsize,:])
    u  = sigm.(gates[1+2hsize:3hsize,:])
    f1 = sigm.(w[5] * h1 .+ w[7])
    f2 = sigm.(w[6] * h2 .+ w[7])
    c  = i .* u .+ f1 .* c1 .+ f2 .* c2
    h  = o .* tanh.(c)
    return (h,c)
end

let
    global traverse
    function traverse(w, tree)
        h,c,hs,ys = helper(w,tree, Any[], Any[])
        return hs,ys
    end

    function helper(w,t,hs,ys)
        h = c = nothing
        if length(t.children) == 1 && isleaf(t.children[1])
            l = t.children[1]
            h,c = lstm(w,l.data)
        elseif length(t.children) == 2
            t1,t2 = t.children[1], t.children[2]
            h1,c1,hs,ys = helper(w,t1,hs,ys)
            h2,c2,hs,ys = helper(w,t2,hs,ys)
            h,c = slstm(w,h1,h2,c1,c2)
        else
            error("invalid tree")
        end
        return (h,c,[hs...,h],[ys...,t.data])
    end
end

# treenn loss function
function loss(w, tree, values=[])
    hs, ygold = traverse(w, tree)
    ypred = w[end-1] * hcat(hs...)
    len = length(ygold)
    lossval = nll(ypred,ygold; average=false)
    push!(values, lossval); push!(values, len)
    return lossval/len
end

# tag given input sentence
function predict(w,tree)
    total = 0
    hs, ys = traverse(w, tree)
    ypred = w[end-1] * hs[end]
    ypred = convert(Array{Float32}, ypred)[:]
    return (indmax(ypred),length(ys))
end

lossgradient = grad(loss)

function train!(w,tree,opt)
    values = []
    gloss = lossgradient(w, tree, values)
    update!(w,gloss,opt)
    return (values...)
end

splitdir(PROGRAM_FILE)[end] == "treenn.jl" && main(ARGS)

end # module
