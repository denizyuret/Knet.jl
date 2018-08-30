using ZipFile

"Where to download dataset from"
const TREEBANK_URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"

"Where to download dataset to"
const TREEBANK_DIR = joinpath(@__DIR__,"treebank")

const TREEBANK_ZIPNAME = "trainDevTestTrees_PTB.zip"
const TREEBANK_SPLITS = ("train", "dev", "test")
const UNK = "_UNK_"
"""

This utility loads [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html)
sentiment classification dataset. There are 8544, 1101 and 2210 trees in train,
dev and test splits respectively. Each tree represented as a s-expression in raw
data. There are 5 different scales to represent sentiment negativity/positivity.

```
# Usage:
include(Pkg.dir("Knet/data/treebank.jl"))
trn = load_treebank_data(splits=["train"]) # possible splits: train,dev,test
trn = load_treebank_data("train")

# each instance is a SentimentTree
# typeof(trn[1]) == SentimentTree

# build word/tag vocabularies
l2i, w2i, i2l, i2w = build_treebank_vocabs(trn)

# make_data! overwrites the data - replaces words/tags with assigned indices
make_data!(trn, w2i, l2i)

# see the data loader source code and the example for further documentation
```

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
    function load_treebank_data(datadir=TREEBANK_DIR, splits=["train","dev"])
        datadir = abspath(datadir)
        if !isdatadir(datadir)
            mkpath(datadir)
        end

        if !isdata(datadir)
            !iszip(datadir) && download_zip(datadir)
            extract_files(datadir)
        end

        data = map(s->load_treebank_data(datadir, s), splits)
    end

    function load_treebank_data(datadir, split::T) where T <: String
        !in(split, TREEBANK_SPLITS) && error("no such split as $split")
        filename = split*".txt"
        filepath = joinpath(datadir, filename)
        data = read_file(filepath)
    end

    function extract_files(datadir)
        r = ZipFile.Reader(joinpath(datadir, TREEBANK_ZIPNAME))
        for f in r.files
            _, this_file = splitdir(f.name)
            split, _ = splitext(this_file)
            if split in TREEBANK_SPLITS
                lines = readlines(f)
                text = join(lines, "\n")
                file = joinpath(datadir, split*".txt")
                open(file, "w") do f
                    write(f, text)
                end
            end
        end
        close(r)
    end

    function isdata(datadir)
        for split in TREEBANK_SPLITS
            splitfile = joinpath(datadir, split*".txt")
            !isfile(splitfile) && return false
        end
        return true
    end

    function iszip(datadir)
        fullpath = joinpath(datadir, TREEBANK_ZIPNAME)
        return isfile(fullpath)
    end

    function download_zip(datadir)
        dest = joinpath(datadir, TREEBANK_ZIPNAME)
        download(TREEBANK_URL, dest)
    end

    function isdatadir(datadir)
        return isdir(datadir)
    end

    function read_file(file)
        data = open(file, "r") do f
            map(parse_line, readlines(f))
        end
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
