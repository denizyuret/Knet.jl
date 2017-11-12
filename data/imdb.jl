using PyCall,JSON

"IMDB Movie reviews sentiment classification dataset from https://keras.io/datasets"
function imdb(;
              url = "https://s3.amazonaws.com/text-datasets",
              dir = Pkg.dir("Knet","data"),
              data="imdb.npz",
              dict="imdb_word_index.json"
              )
    global _imdb_xtrn,_imdb_ytrn,_imdb_xtst,_imdb_ytst,_imdb_dict
    if !isdefined(:_imdb_xtrn)
        datapath = joinpath(dir,data)
        dictpath = joinpath(dir,dict)
        isfile(datapath) || download("$url/$data",datapath)
        isfile(dictpath) || download("$url/$dict",dictpath)
        @pyimport numpy
        d = numpy.load(datapath)
        _imdb_xtrn = map(numpy.asarray, get(d, "x_train"))
        _imdb_ytrn = get(d, "y_train")
        _imdb_xtst = map(numpy.asarray, get(d, "x_test"))
        _imdb_ytst = get(d, "y_test")
        _imdb_dict = JSON.parsefile(dictpath)
    end
    return _imdb_xtrn,_imdb_ytrn,_imdb_xtst,_imdb_ytst,_imdb_dict
end
