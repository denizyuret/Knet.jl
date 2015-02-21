using InplaceOps

function backprop(net::Net, x, y, loss=softmaxloss)
    x = forw(net, x)
    loss(x, y)
    back(net, y)
end

function train(net::Net, x, y; batch=128, iters=0, loss=softmaxloss)
    inittrain(net)
    xrows,xcols = size(x)
    yrows,ycols = size(y)
    for b = 1:batch:xcols
        e = b + batch - 1
        if (e > xcols || b == 1)
            (e > xcols) && (e = xcols)
            xx = similar(net[1].w, (xrows, e-b+1))
            yy = similar(net[1].w, (yrows, e-b+1))
        end
        copy!(xx, (1:xrows,1:e-b+1), x, (1:xrows,b:e))
        copy!(yy, (1:yrows,1:e-b+1), y, (1:yrows,b:e))
        backprop(net, xx, yy, loss)
        for l in net
            isdefined(l,:w) && update(l.w, l.dw, l.pw)
            isdefined(l,:b) && update(l.b, l.db, l.pb)
        end
        iters > 0 && e/batch >= iters && break
    end
end

function predict(net::Net, x; batch=0)
    xrows,xcols = size(x)
    yrows,ycols = size(net[end].w, 1), xcols
    y = similar(x, (yrows, ycols))
    (batch == 0) && (batch = xcols)
    for b = 1:batch:xcols
        e = b + batch - 1
        if (e > xcols || b == 1)
            (e > xcols) && (e = xcols)
            xx = similar(net[1].w, (xrows, e-b+1))
        end
        yy = copy!(xx, (1:xrows,1:e-b+1), x, (1:xrows,b:e))
        yy = forw(net, yy, false)
        copy!(y, (1:yrows,b:e), yy, (1:yrows,1:e-b+1))
    end
    return y
end

function forw(l::Layer, x, apply_fx=true)
    initforw(l, x)
    isdefined(l,:fx) && apply_fx && l.fx(l,x)
    @into! l.y = l.w * x
    isdefined(l,:b) && (@in1! l.y .+ l.b)
    isdefined(l,:fy) && l.fy(l,l.y)
    l.x = x
    return l.y
end

function back(l::Layer, dy, return_dx=true)
    initback(l, dy, return_dx)
    isdefined(l,:fy) && l.fy(l,l.y,dy)
    @into! l.dw = dy * l.x'
    isdefined(l,:b) && sum!(l.db, dy)
    return_dx || return
    @into! l.dx = l.w' * dy
    isdefined(l,:fx) && l.fx(l,l.x,l.dx)
    return l.dx
end

forw(n::Net, x, fx=true) = (for l=n x=forw(l,x,fx) end; x)
back(n::Net, y) = (for i=length(n):-1:1 y=back(n[i],y,i>1) end)
initforw(l, x)=resize(l, :y, l.w, (size(l.w,1),size(x,2)))
initback(l, dy, return_dx)=(resize(l, :dw, l.w); resize(l, :db, l.b); return_dx && resize(l, :dx, l.x))
resize(l, n, a, dims=size(a))=((!isdefined(l,n) || size(l.(n)) != dims) && (l.(n) = similar(a, dims)))

function inittrain(n::Net)
    for l in n
        isdefined(l,:w) && !isdefined(l,:pw) && (l.pw = UpdateParam())    
        isdefined(l,:b) && !isdefined(l,:pb) && (l.pw = UpdateParam())
    end
end
