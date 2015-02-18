using InplaceOps

type Layer w; dw; dw1; dw2; b; db; db1; db2; x; dx; xmask; y; dy; xforw; yforw; xback; yback; Layer()=new() end

function forw(l, x)
    initforw(l, x)
    l.x = x
    # l.xforw(l, l.x)
    @into! l.y = l.w * l.x
    @in1!  l.y .+ l.b
    l.yforw(l.y)
    return l.y
end

function initforw(l, x)
    if (!isdefined(l,:y) || size(l.y) != (size(l.w,1),size(x,2)))
        l.y = similar(l.w, (size(l.w,1),size(x,2)))
    end
end

function back(l, dy, return_dx)
    initback(l, dy, return_dx)
    l.dy = dy
    l.yback(l.y, l.dy)
    @into! l.dw = l.dy * l.x'
    sum!(l.db, l.dy)
    if return_dx
        @into! l.dx = l.w' * l.dy
        # l.dx = l.xback(l, l.dx)
    end
end

function initback(l, dy, return_dx)
    if (!isdefined(l,:dw)) l.dw = similar(l.w) end
    if (!isdefined(l,:db)) l.db = similar(l.b) end
    if (return_dx && (!isdefined(l,:dx) || size(l.dx) != size(l.x)))
        l.dx = similar(l.x) 
    end
end

