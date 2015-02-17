typealias Net Array{Layer,1}

function predict(net, x, batch=128)
    xrows,xcols = size(x)
    yrows,ycols = size(net[end].w, 1), xcols 
    y = similar(x, yrows, ycols)
    xx = similar(net[1].w, xrows, batch)
    for b = 1:batch:xcols
        e = b + batch - 1
        if e > xcols
            e = xcols
            xx = similar(net[1].w, xrows, e-b+1)
        end
        copy!(xx, 1, x, (b-1)*xrows+1, length(xx))
        yy = xx
        for l = 1:length(net)
            yy = forw(net[l], yy)
        end
        copy!(y, (b-1)*yrows+1, yy, 1, length(yy))
    end
    y
end
