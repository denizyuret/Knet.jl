# AbstractLayer: abstract type to collect together different layer
# types and their common operations.

abstract AbstractLayer

# Net: Convenience type for an array of layers

typealias Net Array{AbstractLayer,1}

# Just a convenience type for training etc.
type XY x; y; XY()=new(); end

forw(n::Net, x, fx=true) = (for l=n; x=forw(l,x,fx) end; x)
back(n::Net, dy) = (for i=length(n):-1:1 dy=back(n[i],dy,i>1) end)

function backprop(net::Net, x, dy, loss=softmaxloss)
    y = forw(net, x) 	# y: network output
    loss(y, dy)         # dy: desired output -> gradient
    back(net, dy)       # calculate derivatives
end

function train(net::Net, x, y; batch=128, iters=0, loss=softmaxloss, shuffle=false)
    shuffle && shufflexy!(x,y)
    xrows,xcols = size(x)
    yrows,ycols = size(y)
    (batch == 0 || batch > xcols) && (batch = xcols)
    buf = inittrain(net, x, y, batch)
    for b = 1:batch:xcols
        e = b + batch - 1
        if (e > xcols)
            e = xcols
            chksize(buf, :x, net[1].w, (xrows, e-b+1))
            chksize(buf, :y, net[end].w, (yrows, e-b+1))
        end
        copy!(buf.x, (1:xrows,1:e-b+1), x, (1:xrows,b:e))
        copy!(buf.y, (1:yrows,1:e-b+1), y, (1:yrows,b:e))
        backprop(net, buf.x, buf.y, loss)
        for l in net
            isdefined(l,:w) && update(l.w, l.dw, l.pw)
            isdefined(l,:b) && update(l.b, l.db, l.pb)
        end
        iters > 0 && e/batch >= iters && break
    end
    free(buf.x); free(buf.y) # this should not be necessary now that gc() works...
end

function inittrain(net::Net, x, y, batch)
    for l in net
        isdefined(l,:w) && !isdefined(l,:pw) && (l.pw = UpdateParam())    
        isdefined(l,:b) && !isdefined(l,:pb) && (l.pb = UpdateParam())
    end
    buf = XY()
    chksize(buf, :x, net[1].w, (size(x, 1), batch))
    chksize(buf, :y, net[end].w, (size(y, 1), batch))
    return buf
end

function predict(net::Net, x, y=similar(x, size(net[end].w,1), size(x,2)); batch=0)
    xrows,xcols = size(x)
    yrows,ycols = size(y)
    (batch == 0 || batch > xcols) && (batch = xcols)
    xx = similar(net[1].w, (xrows, batch))
    for b = 1:batch:xcols
        e = b + batch - 1
        if e > xcols
            e = xcols
            batch = e-b+1
            free(xx); xx = similar(net[1].w, (xrows, batch))
        end
        yy = copy!(xx, (1:xrows,1:batch), x, (1:xrows,b:e))
        yy = forw(net, yy, false)
        copy!(y, (1:yrows,b:e), yy, (1:yrows,1:batch))
    end
    free(xx)
    return y
end


function chksize(l, n, a, dims=size(a); fill=nothing)
    if !isdefined(l,n) 
        l.(n) = similar(a, dims)
        fill != nothing && fill!(l.(n), fill)
    elseif size(l.(n)) != dims
        free(l.(n))
        l.(n) = similar(a, dims)
        fill != nothing && fill!(l.(n), fill)
    end
end

function shufflexy!(x, y)
    xrows,xcols = size(x)
    yrows,ycols = size(y)
    @assert xcols == ycols
    x1 = Array(eltype(x), xrows)
    y1 = Array(eltype(y), yrows)
    for n = xcols:-1:2
        r = rand(1:n)
        r == n && continue
        nx = (n-1)*xrows+1; ny = (n-1)*yrows+1
        rx = (r-1)*xrows+1; ry = (r-1)*yrows+1
        copy!(x1, 1, x, nx, xrows)
        copy!(y1, 1, y, ny, yrows)
        copy!(x, nx, x, rx, xrows)
        copy!(y, ny, y, ry, yrows)
        copy!(x, rx, x1, 1, xrows)
        copy!(y, ry, y1, 1, yrows)
    end
end

