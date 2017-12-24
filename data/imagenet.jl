for p in ("MAT","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using MAT,Images

_mcnurl = "http://www.vlfeat.org/matconvnet/models"
_mcndir = Pkg.dir("Knet","data","imagenet")

function matconvnet(name)
    global _mcncache
    if !isdefined(:_mcncache); _mcncache=Dict(); end
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

function imgdata(img, averageImage)
    global _imgcache
    if !isdefined(:_imgcache); _imgcache = Dict(); end
    if !haskey(_imgcache,img)
        if contains(img,"://")
            info("Downloading $img")
            a0 = load(download(img))
        else
            a0 = load(img)
        end
        new_size = ntuple(i->div(size(a0,i)*224,minimum(size(a0))),2)
        a1 = Images.imresize(a0, new_size)
        i1 = div(size(a1,1)-224,2)
        j1 = div(size(a1,2)-224,2)
        b1 = a1[i1+1:i1+224,j1+1:j1+224]
        # ad-hoc solution for Mac-OS image 
        macfix = convert(Array{FixedPointNumbers.Normed{UInt8,8},3}, channelview(b1))
        c1 = permutedims(macfix, (3,2,1))
        d1 = convert(Array{Float32}, c1)
        e1 = reshape(d1[:,:,1:3], (224,224,3,1))
        f1 = (255 * e1 .- averageImage)
        g1 = permutedims(f1, [2,1,3,4])
        _imgcache[img] = g1
    end
    return _imgcache[img]
end

