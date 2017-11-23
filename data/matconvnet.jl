using MAT

_mcnurl = "http://www.vlfeat.org/matconvnet/models"
_mcndir = Pkg.dir("Knet","data","matconvnet")
_mcncache = Dict()

function matconvnet(name)
    if !haskey(_mcncache,name)
        matfile = "$name.mat"
        info("Loading $matfile...")
        path = joinpath(_mcndir,matfile)
        if !isfile(path)
            println("Should I download $matfile?")
            readline()[1] == 'y' || error(:ok)
            isdir(_mcndir) || mkpath(_mcndir)
            download("$_mcnurl/$matfile",path)
        end
        _mcncache[name] = matread(path)
    end
    return _mcncache[name]
end
