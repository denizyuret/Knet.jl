using MAT,Images
export matconvnet, imgdata, make_image_grid

_mcnurl = "http://www.vlfeat.org/matconvnet/models"
_mcndir = joinpath(@__DIR__, "imagenet")

function matconvnet(name)
    global _mcncache
    if !@isdefined(_mcncache); _mcncache=Dict(); end
    if !haskey(_mcncache,name)
        matfile = "$name.mat"
        @info("Loading $matfile...")
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
    if !@isdefined(_imgcache); _imgcache = Dict(); end
    if !haskey(_imgcache,img)
        if occursin("://",img)
            @info("Downloading $img")
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

function make_image_grid(images; gridsize=(8,8), scale=2.0, height=28, width=28)
    shape = (height, width)
    nchannels = size(first(images))[end]
    @assert nchannels == 1 || nchannels == 3
    shp = map(x->Int(round(x*scale)), shape)
    y = map(x->Images.imresize(x,shp), images)
    gridx, gridy = gridsize
    outdims = (gridx*shp[1]+gridx+1,gridy*shp[2]+gridy+1)
    out = zeros(outdims..., nchannels)
    for k = 1:gridx+1; out[(k-1)*(shp[1]+1)+1,:,:] .= 1.0; end
    for k = 1:gridy+1; out[:,(k-1)*(shp[2]+1)+1,:] .= 1.0; end

    x0 = y0 = 2
    for k = 1:length(y)
        x1 = x0+shp[1]-1
        y1 = y0+shp[2]-1
        out[x0:x1,y0:y1,:] .= y[k]

        y0 = y1+2
        if k % gridy == 0
            x0 = x1+2
            y0 = 2
        else
            y0 = y1+2
        end
    end

    out = convert(Array{Float64}, map(x->isnan(x) ? 0 : x, out))
    if nchannels == 1
        out = reshape(out, (size(out,1),size(out,2)))
        out = permutedims(out, (2,1))
    else
        out = permutedims(out, (3,1,2))
    end
    return out
end
