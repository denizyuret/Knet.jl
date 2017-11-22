"Where to download cifar from"
cifarurl = "http://www.cs.toronto.edu/~kriz"

"Where to download cifar to"
cifardir = Pkg.dir("Knet","data","cifar")

"cifar10() => (xtrn,ytrn,xtst,ytst,labels)"
function cifar10(;
                 tgz="cifar-10-binary.tar.gz",
                 dir="cifar-10-batches-bin",
                 trn=["data_batch_$i.bin" for i=1:5],
                 tst=["test_batch.bin"],
                 lbl="batches.meta.txt",
                 )
    global _cifar10_xtrn, _cifar10_ytrn, _cifar10_xtst, _cifar10_ytst, _cifar10_lbls
    if !isdefined(:_cifar10_xtrn)
        _cifar10_xtrn, _cifar10_ytrn, _cifar10_xtst, _cifar10_ytst, _cifar10_lbls = _cifar_read_tgz(tgz,dir,trn,tst,lbl)
    end
    return _cifar10_xtrn, _cifar10_ytrn, _cifar10_xtst, _cifar10_ytst, _cifar10_lbls
end

"cifar100() => (xtrn,ytrn,xtst,ytst,labels)"
function cifar100(;
                  tgz="cifar-100-binary.tar.gz",
                  dir="cifar-100-binary",
                  trn=["train.bin"],
                  tst=["test.bin"],
                  lbl="fine_label_names.txt",
                  )
    global _cifar100_xtrn, _cifar100_ytrn, _cifar100_xtst, _cifar100_ytst, _cifar100_lbls
    if !isdefined(:_cifar100_xtrn)
        _cifar100_xtrn, _cifar100_ytrn, _cifar100_xtst, _cifar100_ytst, _cifar100_lbls = _cifar_read_tgz(tgz,dir,trn,tst,lbl)
    end
    return _cifar100_xtrn, _cifar100_ytrn, _cifar100_xtst, _cifar100_ytst, _cifar100_lbls
end

"Utility to view a cifar image, requires the Images package"
cifarview(x,i)=colorview(RGB,permutedims(x[:,:,:,i],(3,2,1)))

function _cifar_read_tgz(tgz,dir,trn,tst,labels)
    info("Reading $tgz...")
    if !isdir(cifardir)
        mkpath(cifardir)
    end
    dirpath = joinpath(cifardir,dir)
    if !isdir(dirpath)
        tgzpath = joinpath(cifardir,tgz)
        if !isfile(tgzpath)
            url = "$cifarurl/$tgz"
            download(url, tgzpath)
        end
        run(`tar xzf $tgzpath -C $cifardir`)
    end
    xtrn,ytrn = _cifar_read_files(dirpath,trn)
    xtst,ytst = _cifar_read_files(dirpath,tst)
    lbls = readlines(joinpath(dirpath,labels))
    return xtrn,ytrn,xtst,ytst,lbls
end

function _cifar_read_files(dir,files)
    xs,ys = [],[]
    for file in files
        x,y = _cifar_read_file(dir,file)
        push!(xs,x); push!(ys,y)
    end
    return cat(4, xs...), vcat(ys...)
end

function _cifar_read_file(dir,file)
    a = read(joinpath(dir,file))
    d = contains(dir,"cifar-100") ? 1 : 0
    a = reshape(a, (3073+d, div(length(a),3073+d)))
    y = a[1+d,:] + 0x1 # first row (second for cifar100) is Int8 index representation of correct answers
    x = a[2+d:end,:] # rows 2:end (3:end for cifar100) give 32,32,3,N images
    # y = full(sparse(y,1:length(y),1f0,10,length(y))) # one-hot vector representation
    # maybe convert y to int?
    x = reshape(x ./ 255f0, (32,32,3,div(length(x),3072)))
    return (x,y)
end

nothing
