function gradcheck(m::Model, data, loss; gcheck=10, o...)
    rnum = time_ns()
    isdefined(data,:rng) && (data_rng_save = data.rng; data.rng=MersenneTwister(); srand(data.rng,rnum))
    isdefined(m,:rng) && (m_rng_save = m.rng; m.rng=MersenneTwister(); srand(m.rng,rnum))
    l = zeros(2)
    train(m, data, loss; gcheck=true, losscnt=fill!(l,0), o...)
    loss0 = l[1]
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
            isdefined(data,:rng) && srand(data.rng, rnum)
            isdefined(m,:rng) && srand(m.rng, rnum)
            train(m, data, loss; gcheck=true, losscnt=fill!(l,0), o...)
            loss1 = l[1]
            p.out[i] = wi0
            dwi = (loss1 - loss0) / (wi1 - wi0)
            dw0 = pdiff[n][i]
            z=abs(dw0-dwi)-rtol*max(abs(dw0),abs(dwi)); z>maxbad && (maxbad=z)
            if !isapprox(dw0, dwi; rtol=rtol, atol=atol)
                println("gc:fail@$n/$i/$wlen wi:$wi0->$wi1 loss:$loss0->$loss1 dw_bprop=$dw0 dw_numeric=$dwi")
            end
        end
        @assert isequal(p.out, psave)
    end
    isdefined(data,:rng) && (data.rng = data_rng_save)
    isdefined(m,:rng) && (m.rng = m_rng_save)
    println("gc:atol=$maxbad for rtol=$rtol, delta=$delta for largest $(length(pdiff))x$(gcheck) gradients: abs(x-y) <= atol+rtol*max(abs(x),abs(y))")
end


# Why loss0 = l[1], instead of l[1]/l[2]?
# gcheck=true ensures we process one item/sequence. In case of item
# l[2]=1 so it doesn't matter.  In case of a sequence l[2]=tokencnt,
# but we want the total loss, because that is what determined dw.
