# This assumes the structure x[i][t][d] and y[i][t][d]
# where i: instance, t: time, d: dimension
# y[i][t] can be 'nothing'

# TODO:
# - can x[i][t] be 'nothing'?
# - can x[i][t] be multi-dimensional?
# - how do we handle non-sequence inputs?
# - how do we handle contiguous inputs?

function train(r::RNN, x, y)
    err = 0
    for i=1:length(x)
        err += backprop(r, x[i], y[i])
        update(r) # t:39
    end
    return err/length(x)
end
    
function test(r::RNN, x, y)
    err = 0
    for i=1:length(x)
        err += tforw(r, x[i], y[i]; train=false)
    end
    return err/length(x)
end

function backprop(r::RNN, xi, yi) # t:5680
    err = tforw(r, xi, yi; train=true) # t:2952
    tback(r, yi) # t:2648
    return err
end

function tforw(r::RNN, xi, yi; train=true)
    err = 0
    init(r, xi[1]) # t:61
    for t=1:length(xi)
        forw(r, xi[t]; train=train) # t:2952
        err += loss(r, yi[t])       # t:17
    end
    return err
end

function tback(r::RNN, yi)
    for t=length(yi):-1:1
        back(r, yi[t]) # t:2648 TODO: if we take xi as a parameter here maybe the net would not have to remember it?
    end
end

function gradcheck(r::RNN, xi, yi; delta=1e-4, rtol=eps(Float64)^(1/5), atol=eps(Float64)^(1/5), ncheck=10)
    backprop(r, xi, yi)
    dw = cell(nops(r))
    for n=1:nops(r)
        p = param(r.op[n])
        p == nothing && continue
        dw[n] = convert(Array, p.diff)
    end
    for n=1:nops(r)
        p = param(r.op[n])
        p == nothing && continue
        w = convert(Array, p.arr)
        wlen = length(w)
        irange = (wlen <= ncheck ? (1:wlen) : rand(1:wlen, ncheck))
        for i in irange
            wi0 = w[i]
            wi1 = wi0 - delta
            wi2 = wi0 + delta
            # Do not cross 0 for softloss
            (wi0>=0) && (wi1<0) && (wi1=0)
            (wi0<0) && (wi2>0) && (wi2=0)
            w[i] = wi1; copy!(p.arr, w); loss1 = tforw(r, xi, yi; train=false)
            w[i] = wi2; copy!(p.arr, w); loss2 = tforw(r, xi, yi; train=false)
            w[i] = wi0; copy!(p.arr, w)
            dwi = (loss2 - loss1) / (wi2 - wi1)
            if !isapprox(dw[n][i], dwi; rtol=rtol, atol=atol)
                return (n, i, dw[n][i], dwi)
            end
        end
    end
end

function train2(r::RNN, x, y)
    err = 0
    gnorm = zeros(nops(r))
    for i=1:length(x)
        err += backprop(r, x[i], y[i])
        for j=1:nops(r)         
            p=param(r.op[j])
            p==nothing && continue
            v=vecnorm(p.diff)
            v > gnorm[j] && (gnorm[j]=v)
            # v > 100 && scale!(100/v, p.diff) # DONE: move this to update.jl
        end
        update(r) # t:39
    end
    return (err/length(x), maximum(gnorm))
end
    
# More efficient training for rnn's that have a single output with a final output net
# x is a sequence of tokens, y is the desired final output
# TODO: make this into a model type and have a common model interface
function forwback2(r::RNN, o::RNN, x, y)
    forw2(r, o, x; train=true)
    err = loss(o, y)
    back2(r, o, y, length(x))
    return err
end

function back2(r::RNN, o::RNN, y, nt)
    back(o, y)
    back(r, o.dif[nops(o)+1])
    for t=1:nt-1; back(r, nothing); end
end

function forw2(r::RNN, o::RNN, x; a...)
    n = nops(r)
    forw(r, x; a...)
    forw(o, r.out[n:n]; a...)   # passing a vector for x forces init
end

function back2a(r::RNN, o::RNN, xi, yi)
    nt = length(xi)
    dy = back(o, yi) # TODO: this doesn't work because back does not return anything yet
    dy = o.dif[nops(o)+1]
    back(r, dy)
    for t=nt-1:-1:1; back(r, nothing); end
end

function forw2a(r::RNN, o::RNN, xi, yi; a...)
    init(r, xi[1])
    nt = length(xi)
    for t=1:nt; forw(r, xi[t]; a...); end
    ry = r.out[nops(r)]
    init(o, ry)
    forw(o, ry; a...)
end

function forwback1(r::RNN, xi, yi; a...)
    init(r, xi[1])
    nt = length(xi)
    for t=1:nt; forw(r, xi[t]; a...); end
    er = loss(r, yi)
    back(r, yi)
    for t=nt-1:-1:1; back(r, nothing; a...); end
    return er
end

function loss2(r::RNN, o::RNN, x, y)
    forw2(r, o, x; train=false)
    loss(o, y)
end

function grad2(r::RNN, o::RNN, x, y; delta=1e-4, rtol=eps(Float64)^(1/5), atol=eps(Float64)^(1/5), ncheck=10)
    forwback2(r, o, x, y)
    dw = cell(nops(r)+nops(o))
    for n=1:length(dw)
        p = (n<=nops(r) ? param(r.op[n]) : param(o.op[n-nops(r)]))
        dw[n] = (p == nothing ? nothing : convert(Array, p.diff))
    end
    for n=1:length(dw)
        dw[n] == nothing && continue
        p = (n<=nops(r) ? param(r.op[n]) : param(o.op[n-nops(r)]))
        w = convert(Array, p.arr)
        wlen = length(w)
        irange = (wlen <= ncheck ? (1:wlen) : rand(1:wlen, ncheck))
        for i in irange
            wi0 = w[i]
            wi1 = wi0 - delta
            wi2 = wi0 + delta
            # Do not cross 0 for softloss
            (wi0>=0) && (wi1<0) && (wi1=0)
            (wi0<0) && (wi2>0) && (wi2=0)
            w[i] = wi1; copy!(p.arr, w); l1 = loss2(r, o, x, y)
            w[i] = wi2; copy!(p.arr, w); l2 = loss2(r, o, x, y)
            w[i] = wi0; copy!(p.arr, w)
            dwi = (l2 - l1) / (wi2 - wi1)
            if !isapprox(dw[n][i], dwi; rtol=rtol, atol=atol)
                return (n, i, dw[n][i], dwi)
            end
        end
    end
end
