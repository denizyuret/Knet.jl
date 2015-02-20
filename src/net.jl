typealias Net Array{Layer,1}

function train(net::Net, x, y, o=TrainOpts(); args...)
    inittrain(net, x, y, o, args)
    xrows,xcols = size(x)
    yrows,ycols = size(y)
    for b = 1:o.batch:xcols
        e = b + o.batch - 1
        if (e > xcols || b == 1)
            if (e > xcols) e = xcols end
            xbuf = similar(net[1].w, (xrows, e-b+1))
            ybuf = similar(net[1].w, (yrows, e-b+1))
        end
        xtmp = copy!(xbuf, (1:xrows,1:e-b+1), x, (1:xrows,b:e))
        ytmp = copy!(ybuf, (1:yrows,1:e-b+1), y, (1:yrows,b:e))
        for l=1:length(net)     
            if (o.dropout > 0) dropforw(net[l], xtmp, o.dropout) end
            xtmp = forw(net[l], xtmp) 
        end
        for l=length(net):-1:2
            ytmp = back(net[l], ytmp)
            if (o.dropout > 0) dropback(net[l], ytmp, o.dropout) end
        end
        back(net[1], ytmp, false)
        for l=1:length(net)
            update(net[l], o) 
        end
        if (o.iters > 0 && e/o.batch >= o.iters) break end
    end
end

function predict(net::Net, x, batch=0)
    xrows,xcols = size(x)
    yrows,ycols = size(net[end].w, 1), xcols 
    y = similar(x, (yrows, ycols))
    if (batch == 0) batch = xcols end
    for b = 1:batch:xcols
        e = b + batch - 1
        if (e > xcols || b == 1)
            if (e > xcols) e = xcols end
            xx = similar(net[1].w, (xrows, e-b+1))
        end
        yy = copy!(xx, (1:xrows,1:e-b+1), x, (1:xrows,b:e))
        for l = 1:length(net)
            yy = forw(net[l], yy)
        end
        copy!(y, (1:yrows,b:e), yy, (1:yrows,1:e-b+1))
    end
    y
end

function backprop(net::Net, x, y)
    for l = 1:length(net)
        x = forw(net[l], x)
    end
    for l = length(net):-1:1
        y = back(net[l], y, (l>1))
    end
end

function inittrain(net::Net, x, y, o::TrainOpts, args)
    for (a,v) in args 
        if isdefined(o,a) 
            o.(a) = v 
        else
            warn("train: Ignoring unrecognized option $a")
        end
    end
end

