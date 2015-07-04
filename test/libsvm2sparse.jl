using KUnet: hcat!

SMAX = (1<<30)

function libsvm2sparse(fname)
    f = open(fname)
    x = spzeros(Float32,SMAX,0)
    y = zeros(Float32,2,0)
    for l in eachline(f)
        a = split(l)
        y1 = zeros(Float32,2,1)
        y1[a[1]=="1" ? 1 : 2] = 1
        x1 = spzeros(Float32,SMAX,1)
        for i=2:length(a)
            (idx,val) = map(int,split(a[i],':'))
            @assert val==1 (idx,val)
            @assert idx<=SMAX
            x1[idx,1] = 1
        end
        x = hcat!(x, x1, [1], 1)
        y = hcat!(y, y1, [1], 1)
        # x = [x x1]
        # y = [y y1]
    end
    x.m = maximum(x.rowval)
    close(f)
    return (x,y)
end
