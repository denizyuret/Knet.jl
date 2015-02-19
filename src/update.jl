type TrainOpts adagrad; batch; dropout; iters; l1reg; l2reg; learningRate; maxnorm; momentum; nesterov; 
TrainOpts()=new(0f0,    128,   0f0,     0,     0f0,   0f0,   0.01f0,       0f0,     0f0,      false) end

function update(l::Layer, o::TrainOpts)
    initupdate(l, o)
    if o.l1reg > 0
        l.dw += o.l1reg * sign(w)
    end
    if o.l2reg > 0
        l.dw += o.l2reg * l.w
    end
    if o.adagrad > 0
        l.dw2 += l.dw .* l.dw
        l.db2 += l.db .* l.db
        l.dw /= o.adagrad + sqrt(l.dw2)
        l.db /= o.adagrad + sqrt(l.db2)
    end
    if o.learningRate != 1.0
        @in1! l.dw .* o.learningRate
        @in1! l.db .* o.learningRate
    end
    if o.momentum > 0
        l.dw1 = o.momentum * l.dw1 + l.dw
        l.db1 = o.momentum * l.db1 + l.db
        if o.nesterov
            l.dw += o.momentum * l.dw1
            l.db += o.momentum * l.db1
        else
            l.dw = l.dw1
            l.db = l.db1
        end
    end
    @in1! l.w .- l.dw
    @in1! l.b .- l.db
    if o.maxnorm > 0
        norms = sqrt(sum(w.^2, 2))
        if any(norms > o.maxnorm)
            scale = min(o.maxnorm ./ norms, 1)
            l.w *= scale
        end
    end
end

function initupdate(l, o)
    if o.adagrad > 0
        if (!isdefined(l,:dw2)) l.dw2 = zeros(l.dw) end
        if (!isdefined(l,:db2)) l.db2 = zeros(l.db) end
    end
    if o.momentum > 0
        if (!isdefined(l,:dw1)) l.dw1 = zeros(l.dw) end
        if (!isdefined(l,:db1)) l.db1 = zeros(l.db) end
    end
end
