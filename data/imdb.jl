# Based on https://github.com/fchollet/keras/raw/master/keras/datasets/imdb.py
# Also see https://github.com/fchollet/keras/raw/master/examples/imdb_lstm.py
# Also see https://github.com/ilkarman/DeepLearningFrameworks/raw/master/common/utils.py

for p in ("PyCall","JSON","JLD")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using PyCall,JSON,JLD

"""

    imdb()

Load the IMDB Movie reviews sentiment classification dataset from
https://keras.io/datasets and return (xtrn,ytrn,xtst,ytst,dict) tuple.

# Keyword Arguments:
- url=https://s3.amazonaws.com/text-datasets: where to download the data (imdb.npz) from.
- dir=Pkg.dir("Knet/data"): where to cache the data.
- maxval=nothing: max number of token values to include. Words are ranked by how often they occur (in the training set) and only the most frequent words are kept. nothing means keep all, equivalent to maxval = vocabSize + pad + stoken.
- maxlen=nothing: truncate sequences after this length. nothing means do not truncate.
- seed=0: random seed for sample shuffling. Use system seed if 0.
- pad=true: whether to pad short sequences (padding is done at the beginning of sequences). pad_token = maxval.
- stoken=true: whether to add a start token to the beginning of each sequence. start_token = maxval - pad.
- oov=true: whether to replace words >= oov_token with oov_token (the alternative is to skip them). oov_token = maxval - pad - stoken.

"""
function imdb(;
              url = "https://s3.amazonaws.com/text-datasets",
              dir = Pkg.dir("Knet","data","imdb"),
              data="imdb.npz",
              dict="imdb_word_index.json",
              jld="imdb.jld",
              maxval=nothing,
              maxlen=nothing,
              seed=0, oov=true, stoken=true, pad=true
              )
    global _imdb_xtrn,_imdb_ytrn,_imdb_xtst,_imdb_ytst,_imdb_dict
    if !isdefined(:_imdb_xtrn)
        isdir(dir) || mkpath(dir)
        jldpath = joinpath(dir,jld)
        if !isfile(jldpath)
            info("Downloading IMDB...")
            datapath = joinpath(dir,data)
            dictpath = joinpath(dir,dict)
            isfile(datapath) || download("$url/$data",datapath)
            isfile(dictpath) || download("$url/$dict",dictpath)
            @pyimport numpy as np
            d = np.load(datapath)
            _imdb_xtrn = map(a->np.asarray(a,dtype=np.int32), get(d, "x_train"))
            _imdb_ytrn = Array{Int8}(get(d, "y_train") .+ 1)
            _imdb_xtst = map(a->np.asarray(a,dtype=np.int32), get(d, "x_test"))
            _imdb_ytst = Array{Int8}(get(d, "y_test") .+ 1)
            _imdb_dict = Dict{String,Int32}(JSON.parsefile(dictpath))
            JLD.@save jldpath _imdb_xtrn _imdb_ytrn _imdb_xtst _imdb_ytst _imdb_dict
            #rm(datapath)
            #rm(dictpath)
        end
        info("Loading IMDB...")
        JLD.@load jldpath _imdb_xtrn _imdb_ytrn _imdb_xtst _imdb_ytst _imdb_dict
    end
    if seed != 0; srand(seed); end
    xs = [_imdb_xtrn;_imdb_xtst]
    if maxlen == nothing; maxlen = maximum(map(length,xs)); end
    if maxval == nothing; maxval = maximum(map(maximum,xs)) + pad + stoken; end
    if pad; pad_token = maxval; maxval -= 1; end
    if stoken; start_token = maxval; maxval -= 1; end
    if oov; oov_token = maxval; end
    function _imdb_helper(x,y)
        rp = randperm(length(x))
        newy = y[rp]
        newx = similar(x)
        for i in 1:length(x)
            xi = x[rp[i]]
            if oov
                xi = map(w->(w<=oov_token ? w : oov_token), xi)
            else
                xi = filter(w->(w<=oov_token), xi)
            end
            if stoken
                xi = [ start_token; xi ]
            end
            if length(xi) > maxlen
                xi = xi[end-maxlen+1:end]
            end
            if pad && length(xi) < maxlen
                xi = append!(repmat([pad_token], maxlen-length(xi)), xi)
            end
            newx[i] = xi
        end
        newx,newy
    end
    xtrn,ytrn = _imdb_helper(_imdb_xtrn,_imdb_ytrn)
    xtst,ytst = _imdb_helper(_imdb_xtst,_imdb_ytst)
    return xtrn,ytrn,xtst,ytst,_imdb_dict
end

