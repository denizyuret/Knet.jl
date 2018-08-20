using Knet
const WIKINER_DOWNLOAD_PREFIX =
    "https://github.com/neulab/dynet-benchmark/raw/master/data/tags/"
const WIKINER_DIR = Knet.dir("data","wikiner")
const WIKINER_FILES = ("train.txt","dev.txt")
const UNK = "_UNK_"
const PAD = "<*>"

"""

This utility loads [WikiNER](https://github.com/neulab/dynet-benchmark/tree/master/data/tags)
dataset, a named entity recognition dataset used for DyNet benchmarking.
There are 142153 and 1696 sentences in train and dev splits respectively. This
data has 119102 different words and 9 different tags.

```
# Usage:
include(Pkg.dir("Knet/data/wikiner.jl"))
data = WikiNERData()

# data is a struct and it has the following fields,
# - trn: training data - Array{Any,1}
# - dev: validation data - Array{Any,1}
# - words: all words in the training data
# - tags: all tags - integers
# - wordcounts: word count dictionary
# - nwords: number of different words in corpus
# - ntags: number of tags
# - nchars: number of characters - actually bytes
# - w2i: word2index dictionary
# - i2w: index2word dictionary
# - t2i: tag2index dictionary
# - i2t: index2tag dictionary
# - c2i: char2index dictionary
# - i2c: index2char dictionary
```

"""
mutable struct WikiNERData
    trn
    dev
    words
    tags
    wordcounts
    nwords
    ntags
    nchars
    w2i
    i2w
    t2i
    i2t
    c2i
    i2c

    function WikiNERData(datadir=WIKINER_DIR,minoccur=6)
        (trn,dev,words,tags,chars,wc,fwords,vocab) =
            load_wikiner_data(datadir,minoccur)
        w2i, i2w, t2i, i2t, c2i, i2c = vocab
        nwords = length(wc)
        ntags = length(t2i)
        nchars = length(c2i)

        return new(
            trn,dev,words,tags,wc,nwords,ntags,nchars,w2i,i2w,t2i,i2t,c2i,i2c)
    end
end

let
    global load_wikiner_data
    function load_wikiner_data(datadir,minoccur)
        trn,dev = read_wikiner_data(datadir)
        words, tags, chars = get_words_tags_chars(trn)
        wordcounts = count_words(words)
        fwords = filter_words(wordcounts,minoccur)
        vocab = build_vocabularies(fwords, tags, chars)
        return trn, dev, words, tags, chars, wordcounts, fwords, vocab
    end

    function read_wikiner_data(datadir, splits=["train","dev"])
        make_wikiner_data(datadir)
        data = map(s->read_wikiner_data(datadir,s), splits)
    end

    function read_wikiner_data(datadir, split::T) where T <: String
        filename = split*".txt"
        filepath = joinpath(datadir, filename)
        !isfile(filepath) && error("no such split/file in the dataset")
        data = open(filepath, "r") do f
            map(parse_line, readlines(f))
        end
    end

    function make_wikiner_data(datadir=WIKINER_DIR)
        !isdir(datadir) && mkpath(datadir)
        for filename in WIKINER_FILES
            download_url = WIKINER_DOWNLOAD_PREFIX*filename
            destination = joinpath(datadir, filename)
            !isfile(destination) && download(download_url, destination)
        end
    end

    function parse_line(line)
        return map(x->split(x,"|"), split(replace(line,"\n"=>""), " "))
    end

    function get_words_tags_chars(trn)
        words, tags, chars = [], [], Set()
        for sample in trn
            for (word,tag) in sample
                push!(words, word)
                push!(tags, tag)
                push!(chars, convert(Array{UInt8,1}, codeunits(word))...)
            end
        end
        push!(chars, PAD)
        return words, tags, chars
    end

    function filter_words(wordcounts,minoccur)
        filtered_words = filter(x-> x[2] >= minoccur, wordcounts)
        filtered_words = collect(keys(filtered_words))
        !in(UNK, filtered_words) && push!(filtered_words, UNK)
        return filtered_words
    end

    function count_words(words)
        wordcounts = Dict()
        for word in words
            wordcounts[word] = get(wordcounts, word, 0) + 1
        end
        return wordcounts
    end

    function build_vocabularies(words, tags, chars)
        w2i, i2w = build_vocabulary(words)
        t2i, i2t = build_vocabulary(tags)
        c2i, i2c = build_vocabulary(chars)
        return w2i, i2w, t2i, i2t, c2i, i2c
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
end
