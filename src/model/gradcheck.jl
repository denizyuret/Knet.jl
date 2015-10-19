function gradcheck(m::Model, data, loss; gcheck=10, o...)
    l = zeros(2)
    train(m, data, loss; gcheck=true, losscnt=l)
    loss0 = l[1]/l[2]
    param = params(m)
    pdiff = map(p->convert(Array, p.dif), param)
    delta = atol = rtol = cbrt(eps(eltype(pdiff[1])))  # 6e-6 for Float64, 5e-3 for Float32 works best
    maxbad = 0
    for n=1:length(param)
        p = param[n]
        psave = copy(p.out)
        wlen = length(p.out)
        irange = (wlen <= gcheck ? (1:wlen) : sortperm(abs(vec(pdiff[n])),rev=true)[1:gcheck]) # rand(1:wlen, gcheck))
        for i in irange
            wi0 = p.out[i]
            wi1 = (wi0 >= 0 ? wi0 + delta : wi0 - delta)
            p.out[i] = wi1
            loss1 = test(m, data, loss; gcheck=true)
            p.out[i] = wi0
            dwi = (loss1 - loss0) / (wi1 - wi0)
            dw0 = pdiff[n][i]
            z=abs(dw0-dwi)-rtol*max(abs(dw0),abs(dwi)); z>maxbad && (maxbad=z)
            if !isapprox(dw0, dwi; rtol=rtol, atol=atol)
                println(tuple(:gc, n, i, dw0, dwi))
            end
        end
        @assert isequal(p.out, psave)
    end
    info("gradient checked the largest $(length(pdiff))x$(gcheck) parameters:")
    info("abs(x-y) <= atol+rtol*max(abs(x),abs(y))) where rtol=$rtol, atol=$maxbad")
end
