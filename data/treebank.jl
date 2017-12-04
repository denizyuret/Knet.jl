for p in ("ZipFile",)
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using ZipFile

"Where to download dataset from"
const TREEBANK_URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"

"Where to download dataset to"
const TREEBANK_DIR = Pkg.dir("Knet","data","treebank")

const TREEBANK_ZIPNAME = "trainDevTestTrees_PTB.zip"
const TREEBANK_ZIPPATH = joinpath(TREEBANK_DIR, TREEBANK_ZIPNAME)

# download data for the first time
!isdir(TREEBANK_DIR) && mkpath(TREEBANK_DIR)
!isfile(TREEBANK_ZIPPATH) && download(TREEBANK_URL, TREEBANK_ZIPPATH)

"""

This utility loads [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) sentiment classification dataset. There are 8544, 1101 and 2210 trees in train, dev and test splits respectively. Each tree represented as a s-expression. There are 5 different scales to represent sentiment negativity/positivity.

```
# Usage:
include(Pkg.dir("Knet/data/treebank.jl"))
dtrn = load_treebank_data(splits=["train"]) # possible splits: train,dev,test
dtrn = load_treebank_data("train")
# typeof(dtrn[1]) == StanfordSentimentTree
"""

mutable struct SentimentTree
    label
    children
    data

    SentimentTree(x) = new(x,nothing,nothing)
    SentimentTree(x,y) = new(x,y,nothing)
end

const SentimentTrees = Array{SentimentTree,1}


function isleaf(t::SentimentTree)
    return t.children == nothing
end

function pretty(t::SentimentTree)
    t.children == nothing && return t.label
    join([t.label; map(pretty, t.children)], " ")
end

let
    items = []

    global leaves
    function leaves(t::SentimentTree)
        empty!(items)
        helper(t)
        return items
    end

    function helper(subtree)
        if isleaf(subtree)
            push!(items, subtree)
        else
            for child in subtree.children
                helper(child)
            end
        end
    end
end

let
    nodes = []
    global nonterms
    function nonterms(t::SentimentTree)
        empty!(nodes)
        helper(t)
        return nodes
    end

    function helper(subtree)
        if !isleaf(subtree)
            push!(nodes, subtree)
            map(helper, subtree.children)
        end
    end
end

# data load function(s)
let
    global load_treebank_data
    function load_treebank_data(splits=["train","dev","test"])
        data = map(load_treebank_data, splits)
    end

    function load_treebank_data(split::T) where T <: String
        filename = split*".txt"
        r = ZipFile.Reader(TREEBANK_ZIPPATH)
        data = nothing
        for f in r.files
            if f.name == "trees/"*filename
                data = map(parse_line, readlines(f))
                break
            end
        end
        close(r)
        data == nothing && error("no such split/file in zip archive")
        return data
    end

    function parse_line(line)
        ln = replace(line, "\n", "")
        tokens = tokenize_sexpr(ln)
        shift!(tokens)
        return within_bracket(tokens)[1]
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
                return SentimentTree(label, children), state
            else
                push!(children, SentimentTree(token))
            end
        end
    end
end

let
    global build_treebank_vocabs
    function build_treebank_vocabs(trees::SentimentTrees)
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
end

function make_data!(trees::SentimentTrees, w2i, l2i)
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
