module VGG

using Knet,Images,MAT

op = [1,2,1,2,1,1,2,1,1,2,1,1,2,3,3,4]

convx(w,x)=conv4(w,x;padding=1,mode=1)

function predict(w,x)
    for k=1:div(length(w),2)
        if op[k] == 1
            x = relu(convx(w[2k-1],x) .+ w[2k])
        elseif op[k] == 2
            x = pool(relu(convx(w[2k-1],x) .+ w[2k]))
        elseif op[k] == 3
            x = relu(w[2k-1]*mat(x) .+ w[2k])
        else
            x = w[2k-1]*mat(x) .+ w[2k]
        end
    end
    return x
end

vgg = matread("imagenet-vgg-verydeep-16.mat")
w = Any[]
for l in vgg["layers"]
    haskey(l,"weights") && push!(w, l["weights"]...)
end
for i in 2:2:26
    w[i] = reshape(w[i], (1,1,length(w[i]),1))
end
for i in 27:2:32
    w[i] = mat(w[i])'
end
w = map(KnetArray,w)

averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"])
description = vgg["meta"]["classes"]["description"]

function classify(img)
    contains(img,"://") && (img = download(img))
    a0 = load(img)
    new_size = ntuple(i->div(size(a0,i)*224,minimum(size(a0))),2)
    a1 = Images.imresize(a0, new_size)
    i1 = div(size(a1,1)-224,2)
    j1 = div(size(a1,2)-224,2)
    b1 = a1[i1+1:i1+224,j1+1:j1+224]'
    c1 = separate(b1)
    d1 = convert(Array{Float32}, c1)
    e1 = reshape(d1[:,:,1:3], (224,224,3,1))
    f1 = (255 * e1 .- averageImage)
    x1 = KnetArray(f1)
    y1 = predict(w,x1)
    z1 = vec(Array(y1))
    s1 = sortperm(z1,rev=true)
    p1 = exp(logp(z1))
    hcat(p1[s1[1:5]], description[s1[1:5]])
end

end # module
