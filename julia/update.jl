type Opts learningRate; momentum; adagrad; nesterov; dropout; maxnorm; l1reg; l2reg; Opts()=new() end

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
