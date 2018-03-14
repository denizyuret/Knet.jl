function predict_q(w, x; nh=1)
    inp = x
    for i=1:nh
        inp = relu.(w["w_$i"] * inp .+ w["b_$i"])
    end
    q = w["w_out"] * inp .+ w["b_out"]
    return q
end

function init_weights(input, hiddens, nout, atype)
    w = Dict()
    inp = input
    for i=1:length(hiddens)
        w["w_$i"] = xavier(hiddens[i], inp)
        w["b_$i"] = zeros(hiddens[i])
        inp = hiddens[i]
    end

    w["w_out"] = xavier(nout, hiddens[end])
    w["b_out"] = zeros(nout, 1)
    
    for k in keys(w)
        w[k] = convert(atype, w[k])
    end
    return w
end

function save_model(w, fname)
    tmp = Dict()
    for k in keys(w)
        tmp[k] = convert(Array{Float32}, w[k])
    end
    save(fname, "model", tmp)
end

function load_model(fname, atype)
    w = load(fname, "model")
    for k in keys(w)
        w[k] = convert(atype, w[k])
    end
    return w
end
