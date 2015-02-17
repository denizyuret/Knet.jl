using InplaceOps

type Layer w; dw; dw1; dw2; b; db; db1; db2; x; dx; xmask; y; dy; xforw; yforw; xback; yback; Layer()=new() end

function forw(l, x)
    initforw(l, x)
    l.x = l.xforw(l, x)
    @into! l.y = l.w * l.x
    @in1!  l.y .+ l.b
    l.yforw(l, l.y)
    l.y
end

function initforw(l, x)
    if (!isdefined(l,:y) ||
        size(l.y,1) != size(l.w,1) ||
        size(l.y,2) != size(x,2))
        l.y = similar(l.w, (size(l.w,1), size(x,2)))
    end
end

function back(l, dy, return_dx)
    l.dy = l.yback(l, dy)
    l.dw = l.dy * l.x'
    l.db = sum(l.dy, 2)
    if return_dx
        l.dx = l.w' * l.dy
        l.dx = l.xback(l, l.dx)
    end
end

function reluforw(l, y)
    for i=1:length(y)
        if (y[i] < 0)
            y[i] = 0
        end
    end
end

noop(l,x)=x
reluback(l, dy)=error("reluback not implemented yet")
softforw(l,y)=y
softback(l,dy)=error("softback not implemented yet")
dropforw(l,x)=error("dropforw not implemented yet")
dropback(l,dx)=error("dropback not implemented yet")
