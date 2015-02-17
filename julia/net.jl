typealias Net Array{Layer,1}

function predict(net, x, batch=128)
    xrows,xcols = size(x)
    yrows,ycols = size(net[end].w, 1), xcols 
    y = similar(x, yrows, ycols)
    for b = 1:batch:xcols
        e = min(xcols, b + batch - 1)
        z = sub(x, 1:xrows, b:e)
        for l = 1:length(net)
            z = forw(net[l], z)
        end
        y[:,b:e] = z
    end
    y
end
