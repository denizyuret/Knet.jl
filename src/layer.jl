using InplaceOps

type Layer w; b; yforw; yback; dw; db; dw1; db1; dw2; db2; x; dx; xmask; y; dy; 
    Layer(w,b,yforw,yback)=new(w,b,yforw,yback)
    Layer()=new() 
end

function forw(l, x)
    initforw(l, x)
    l.x = x
    @into! l.y = l.w * l.x
    @in1!  l.y .+ l.b
    l.y = l.yforw(l, l.y)
    return l.y
end

function back(l, dy, return_dx=true)
    initback(l, dy, return_dx)
    l.dy = l.yback(l, dy)
    @into! l.dw = l.dy * l.x'
    sum!(l.db, l.dy)
    if return_dx
        @into! l.dx = l.w' * l.dy
    end
end

function resize(l, f, a, dims=size(a))
    if (!isdefined(l,f) || size(l.(f)) != dims)
        l.(f) = similar(a, dims)
    end
end

function initforw(l, x)
    resize(l, :y, l.w, (size(l.w,1),size(x,2)))
end

function initback(l, dy, return_dx)
    resize(l, :dw, l.w)
    resize(l, :db, l.b)
    if return_dx resize(l, :dx, l.x) end
end
