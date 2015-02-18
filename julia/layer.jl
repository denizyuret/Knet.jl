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
    if (!isdefined(l,:dx) && return_dx) l.dx = similar(l.x) end
end

function initforw(l, x)
    if (!isdefined(l,:y) ||
        size(l.y,1) != size(l.w,1) ||
        size(l.y,2) != size(x,2))
        l.y = similar(l.w, (size(l.w,1), size(x,2)))
    end
end

function reluforw(y)
    for i=1:length(y)
        if (y[i] < 0)
            y[i] = 0
        end
    end
end

function reluback(y, dy)
    for i=1:length(dy)
        if (y[i] <= 0)
            dy[i] = 0
        end
    end
end

softforw(y)=y

function softback(y, dy)
    # we do softmax here instead of in forw
    # overwriting y from unnormalized log probabilities to normalized probabilities
    # NumericExtensions.softmax!(y,y,1) allocates unnecessary memory
    # dy is a 0-1 matrix of correct answers
    # will overwrite it with the gradient
    # TODO: is this a good interface?
    # TODO: other types of final layers, losses?

    for j=1:size(y,2)
        ymax = y[1,j]
        for i=2:size(y,1)
            if (y[i,j] > ymax)
                ymax = y[i,j]
            end
        end
        ysum = 0
        for i=1:size(y,1)
            y[i,j] = exp(y[i,j] - ymax)
            ysum += y[i,j]
        end
        for i=1:size(y,1)
            y[i,j] /= ysum
            dy[i,j] = (y[i,j] - dy[i,j]) / size(y,2)
        end
    end
end


noop(l,x)=x
dropforw(l,x)=error("dropforw not implemented yet")
dropback(l,dx)=error("dropback not implemented yet")
