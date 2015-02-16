module KUnet

type Layer w; dw; dw1; dw2; b; db; db1; db2; x; dx; xmask; y; dy; xforw!; yforw!; xback!; yback!; Layer()=new() end
type Opts learningRate; momentum; adagrad; nesterov; dropout; maxnorm; l1reg; l2reg; Opts()=new() end

function forw!(l, x)
    l.x = l.xforw!(l, x)
    l.y = l.w * l.x
    broadcast!(+, l.y, l.y, l.b)  # l.y = l.y + l.b
    l.y = l.yforw!(l, l.y)
end

function back!(l, dy, return_dx)
    l.dy = l.yback!(l, dy)
    l.dw = l.dy * l.x'
    l.db = sum(l.dy, 2)
    if return_dx
        l.dx = l.w' * l.dy
        l.dx = l.xback!(l, l.dx)
    end
end

function reluforw!(l, y)
    for i=1:length(y)
        if (y[i] < 0)
            y[i] = 0
        end
    end
    y
end

function noop(l, x)
    x
end

function update!(o, w, dw, dw1, dw2)
    if o.l1reg > 0
        dw += o.l1reg * sign(w)
    end
    if o.l2reg > 0
        dw += o.l2reg * w
    end
    if o.adagrad > 0
        dw2 += dw .* dw
        dw /= o.adagrad + sqrt(dw2)
    end
    if o.learningRate != 1.0
        dw *= o.learningRate
    end
    if o.momentum > 0
        dw1 = o.momentum * dw1 + dw
        if o.nesterov
            dw += o.momentum * dw1
        else
            dw = dw1
        end
    end
    w -= dw
    if o.maxnorm > 0
        norms = sqrt(sum(w.^2, 2))
        if any(norms > o.maxnorm)
            scale = min(o.maxnorm ./ norms, 1)
            w *= scale
        end
    end
end

end # module
