module TreeNN
using Knet
using AutoGrad
using ArgParse

const train_file = "data/trees/train.txt"
const dev_file = "data/trees/dev.txt"
const UNK = "_UNK_"
t00 = now()

function main(args)
    s = ArgParseSettings()
    s.description = "Tree-structured LSTM network in Knet"

    @add_arg_table s begin
        ("--gpu"; action=:store_true; help="use GPU or not")
        ("EMBED_SIZE"; arg_type=Int; help="embedding size")
        ("HIDDEN_SIZE"; arg_type=Int; help="hidden size")
        ("SPARSE"; arg_type=Int; help="sparse update 0/1")
        ("TIMEOUT"; arg_type=Int; help="max timeout")
        ("--train"; default=train_file; help="train file")
        ("--dev"; default=dev_file; help="dev file")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--epochs"; arg_type=Int; default=100; help="epochs")
        ("--minoccur"; arg_type=Int; default=0)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && srand(o[:seed])
    atype = o[:gpu] ? KnetArray{Float32} : Array{Float32}

    # read data
    trn = read_file(o[:train])
    tst = read_file(o[:dev])

    # count words and build vocabulary
    l2i, w2i, i2l, i2w = build_vocabs(trn)
    nwords = length(w2i); nlabels = length(l2i)
    make_data!(trn, w2i, l2i); make_data!(tst, w2i, l2i)

    # build model
    w = initweights(
        atype, o[:HIDDEN_SIZE], length(w2i), length(l2i), o[:EMBED_SIZE])
    s = initstate(atype, o[:HIDDEN_SIZE])
    opt = map(x->Adam(), w)

    # train bilstm tagger
    # println("nwords=$nwords, nlabels=$nlabels"); flush(STDOUT)
    println("startup time: ", Int(now()-t00)*0.001); flush(STDOUT)
    all_time = 0
    sents = 0
    for epoch = 1:o[:epochs]
        closs = 0.0
        cwords = 0
        shuffle!(trn)
        t0 = now()
        for k = 1:length(trn)
            sents += 1
            iter = (epoch-1)*length(trn) + k
            tree = trn[k]
            this_loss, this_words = train!(w,s,tree,opt)
            closs += this_loss
            cwords += this_words

            if iter % 1000 == 0
                @printf("%f\n", closs/cwords); flush(STDOUT)
                closs = 0.0
                cwords = 0
            end
        end
        all_time += Int(now()-t0)*0.001

        good = bad = 0
        for tree in tst
            ind, nwords = predict(w, copy(s), tree)
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

        all_time > o[:TIMEOUT] && return
    end
end

# read file
function read_file(file)
    data = open(file, "r") do f
        map(parse_line, readlines(f))
    end
end

# parse line
function parse_line(line)
    ln = replace(line, "\n", "")
    tokens = tokenize_sexpr(ln)
    shift!(tokens)
    return within_bracket(tokens)[1]
end

type Tree
    label
    children
    data
end

function Tree(x)
    return Tree(x,nothing,nothing)
end

function Tree(x,y)
    return Tree(x,y,nothing)
end

function isleaf(t::Tree)
    return t.children == nothing
end

function pretty(t::Tree)
    t.children == nothing && return t.label
    join([t.label; map(pretty, t.children)], " ")
end

function leaves(t::Tree)
    items = []
    function helper(subtree)
        if isleaf(subtree)
            push!(items, subtree)
        else
            for child in subtree.children
                helper(child)
            end
        end
    end
    helper(t)
    return items
end

function nonterms(t::Tree)
    nodes = []
    function helper(subtree)
        if !isleaf(subtree)
            push!(nodes, subtree)
            map(helper, subtree.children)
        end
    end
    helper(t)
    return nodes
end

function tokenize_sexpr(sexpr)
    tokker = r" +|[()]|[^ ()]+"
    filter(t -> t != " ", matchall(tokker, sexpr))
end

function within_bracket(tokens, state=1)
    (label, state) = next(tokens, state)
    children = []
    while !done(tokens, state)
        (token, state) = next(tokens, state)
        if token == "("
            (child, state) = within_bracket(tokens, state)
            push!(children, child)
        elseif token == ")"
            return Tree(label, children), state
        else
            push!(children, Tree(token))
        end
    end
end

function build_vocabs(trees)
    words = Set()
    labels = Set()
    for tree in trees
        push!(words, map(t->t.label, leaves(tree))...)
        push!(labels, map(t->t.label, nonterms(tree))...)
    end
    push!(words, UNK)
    w2i, i2w = build_vocab(words)
    l2i, i2l = build_vocab(labels)
    return l2i, w2i, i2l, i2w
end

function build_vocab(xs)
    x2i = Dict(); i2x = Dict()
    for (i,x) in enumerate(xs)
        x2i[x] = i
        i2x[i] = x
    end
    return x2i, i2x
end

function make_data!(trees, w2i, l2i)
    for tree in trees
        for leaf in leaves(tree)
            ind = get(w2i, leaf.label, w2i[UNK])
            leaf.data = ind
        end
        for nonterm in nonterms(tree)
            nonterm.data = l2i[nonterm.label]
        end
    end
end

# initialize hidden and cell arrays
function initstate(atype, hidden, batchsize=1)
    return convert(atype, zeros(hidden, batchsize))
end

# initialize all weights of the language model
function initweights(atype, hidden, words, labels, embed, winit=0.01)
    w = Array(Any, 9)
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

function traverse(w, s0, tree)
    atype = typeof(AutoGrad.getval(w[1]))
    function helper(t,hs,ys)
        h = c = nothing
        if length(t.children) == 1 && isleaf(t.children[1])
            l = t.children[1]
            h,c = lstm(w,l.data)
        elseif length(t.children) == 2
            t1,t2 = t.children[1], t.children[2]
            h1,c1,hs,ys = helper(t1,hs,ys)
            h2,c2,hs,ys = helper(t2,hs,ys)
            h,c = slstm(w,h1,h2,c1,c2)
        else
            error("invalid tree")
        end
        return (h,c,[hs...,h],[ys...,t.data])
    end

    h,c,hs,ys = helper(tree, Any[], Any[])
    return hs,ys
end

# treenn loss function
function loss(w, s0, tree, values=[])
    atype = typeof(AutoGrad.getval(w[1]))
    total = 0
    hs, ys = traverse(w, copy(s0), tree)
    for (h,ygold) in zip(hs,ys)
        ypred  = w[end-1] * h
        total += logprob([ygold], ypred)
    end

    push!(values, -total); push!(values, length(ys))
    return -total
end

# tag given input sentence
function predict(w,s0,tree)
    atype = typeof(AutoGrad.getval(w[1]))
    total = 0
    hs, ys = traverse(w, copy(s0), tree)
    ypred = w[end-1] * hs[end]
    ypred = convert(Array{Float32}, ypred)[:]
    return (indmax(ypred),length(ys))
end

lossgradient = grad(loss)

function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = output + nrows*(0:(length(output)-1))
    o1 = logp(ypred,1)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end

function train!(w,s,tree,opt)
    values = []
    gloss = lossgradient(w, copy(s), tree, values)
    for k = 1:length(w)
        update!(w[k], gloss[k], opt[k])
    end
    return values
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE=="knet/treenn.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end # module
