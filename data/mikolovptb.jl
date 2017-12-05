mikolovptburl = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
mikolovptbdir = Pkg.dir("Knet","data","mikolovptb")
mikolovptbtrn = "ptb.train.txt"
mikolovptbdev = "ptb.valid.txt"
mikolovptbtst = "ptb.test.txt"

"""

    mikolovptb()

Read [PTB](https://catalog.ldc.upenn.edu/ldc99t42) text from Mikolov's
[RNNLM](http://www.fit.vutbr.cz/~imikolov/rnnlm) toolkit which has
been lowercased and reduced to a 10K vocabulary size.  Return a tuple
(trn,dev,tst,vocab) where

    trn::Vector{Vector{UInt16}}: 42068 sentences, 887521 words
    dev::Vector{Vector{UInt16}}: 3370 sentences, 70390 words
    tst::Vector{Vector{UInt16}}: 3761 sentences, 78669 words
    vocab::Vector{String}: 9999 unique words

"""
function mikolovptb()
    global _mptb_trn, _mptb_dev, _mptb_tst, _mptb_vocab
    if !isdefined(:_mptb_trn)
        isdir(mikolovptbdir) || mkpath(mikolovptbdir)
        if !isfile(joinpath(mikolovptbdir, "ptb.train.txt"))
            info("Downloading $mikolovptburl")
            tgz = download(mikolovptburl)
            run(`tar --strip-components 3 -C $mikolovptbdir -xzf $tgz ./simple-examples/data/ptb.train.txt ./simple-examples/data/ptb.valid.txt ./simple-examples/data/ptb.test.txt`)
        end
        dict = Dict{String,Int}()
        data = Vector{Vector{UInt16}}[]
        for file in (mikolovptbtrn, mikolovptbdev, mikolovptbtst)
            sentences = Vector{UInt16}[]
            for line in eachline(joinpath(mikolovptbdir,file))
                words = UInt16[]
                for word in split(line)
                    widx = get!(dict, word, 1+length(dict))
                    push!(words, widx)
                end
                push!(sentences, words)
            end
            push!(data, sentences)
        end
        _mptb_trn, _mptb_dev, _mptb_tst = data
        _mptb_vocab = Array{String}(length(dict))
        for (k,v) in dict; _mptb_vocab[v] = k; end
    end
    return _mptb_trn, _mptb_dev, _mptb_tst, _mptb_vocab
end

