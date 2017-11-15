# Based on https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
# Also see https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py

for p in ("PyCall","JSON","JLD2")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using PyCall,JSON,JLD2

"IMDB Movie reviews sentiment classification dataset from https://keras.io/datasets"
function imdb(;
              url = "https://s3.amazonaws.com/text-datasets",
              dir = Pkg.dir("Knet","data"),
              data="imdb.npz",
              dict="imdb_word_index.json",
              jld2="imdb.jld2",
              )
    global _imdb_xtrn,_imdb_ytrn,_imdb_xtst,_imdb_ytst,_imdb_dict
    if !isdefined(:_imdb_xtrn)
        jld2path = joinpath(dir,jld2)
        if !isfile(jld2path)
            info("Downloading IMDB...")
            datapath = joinpath(dir,data)
            dictpath = joinpath(dir,dict)
            isfile(datapath) || download("$url/$data",datapath)
            isfile(dictpath) || download("$url/$dict",dictpath)
            @pyimport numpy as np
            d = np.load(datapath)
            _imdb_xtrn = map(a->np.asarray(a,dtype=np.int32), get(d, "x_train"))
            _imdb_ytrn = Array{Int8}(get(d, "y_train"))
            _imdb_xtst = map(a->np.asarray(a,dtype=np.int32), get(d, "x_test"))
            _imdb_ytst = Array{Int8}(get(d, "y_test"))
            _imdb_dict = Dict{String,Int32}(JSON.parsefile(dictpath))
            JLD2.@save jld2path _imdb_xtrn _imdb_ytrn _imdb_xtst _imdb_ytst _imdb_dict
            #rm(datapath)
            #rm(dictpath)
        end
        info("Loading IMDB...")
        JLD2.@load jld2path _imdb_xtrn _imdb_ytrn _imdb_xtst _imdb_ytst _imdb_dict
    end
    return _imdb_xtrn,_imdb_ytrn,_imdb_xtst,_imdb_ytst,_imdb_dict
end
