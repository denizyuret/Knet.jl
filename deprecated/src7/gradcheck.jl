function gradcheck(m, grad, loss; gcheck=10, _eps=cbrt(eps(eltype(m))), delta=_eps, atol=_eps, rtol=_eps, o...)
    # 6e-6 for Float64, 5e-3 for Float32 works best
    # rnum = 42 # time_ns() #DBG
    # isdefined(data,:rng) && (data_rng_save = data.rng; data.rng=MersenneTwister(); srand(data.rng,rnum))
    # isdefined(m,:rng) && (m_rng_save = m.rng; m.rng=MersenneTwister(); srand(m.rng,rnum))
    # l = zeros(2)
    # train(m, data, loss; gcheck=true, losscnt=fill!(l,0), o...)
    # loss0 = l[1]
    loss0 = loss(m)
    grad(m)
    pp = params(m)
    dcopy(p)=(isa(p.dif,Array) ? copy(p.dif) : convert(Array,p.dif))
    ppdif = map(dcopy, pp) # This may get reset in loss, so make a copy now
    maxbad = 0
    for n = 1:length(pp)
        p = pp[n]
        pdif = ppdif[n]
        psave = copy(p.out)     # DBG
        wlen = length(p.out)
        irange = (wlen <= gcheck ? (1:wlen) : sortperm(abs(vec(pdif)),rev=true)[1:gcheck]) # rand(1:wlen, gcheck))
        for i in irange
            wi0 = p.out[i]
            wi1 = (wi0 >= 0 ? wi0 + delta : wi0 - delta)
            p.out[i] = wi1
            # isdefined(data,:rng) && srand(data.rng, rnum)
            # isdefined(m,:rng) && srand(m.rng, rnum)
            # train(m, data, loss; gcheck=true, losscnt=fill!(l,0), o...)
            # loss1 = l[1]
            loss1 = loss(m)
            p.out[i] = wi0
            dw0 = pdif[i]
            dwi = (loss1 - loss0) / (wi1 - wi0)
            z=abs(dw0-dwi)-rtol*max(abs(dw0),abs(dwi)); z>maxbad && (maxbad=z)
            if !isapprox(dw0, dwi; rtol=rtol, atol=atol)
                println("gc:fail@$(findfirst(params(m),p))/$i/$wlen wi:$wi0->$wi1 loss:$loss0->$loss1 dw_bprop=$dw0 dw_numeric=$dwi")
            end
        end
        @assert isequal(p.out, psave) # DBG
    end
    # isdefined(data,:rng) && (data.rng = data_rng_save)
    # isdefined(m,:rng) && (m.rng = m_rng_save)
    @printf("gc:atol=%g for rtol=%g, delta=%g for largest %d x %d gradients: abs(x-y) <= atol+rtol*max(abs(x),abs(y))\n",
            maxbad, rtol, delta, length(params(m)), gcheck)
    # println("gc:atol=$maxbad for rtol=$rtol, delta=$delta for largest $(length(params(m)))x$(gcheck) gradients: abs(x-y) <= atol+rtol*max(abs(x),abs(y))")
end



# Why loss0 = l[1], instead of l[1]/l[2]?
# gcheck=true ensures we process one item/sequence. In case of item
# l[2]=1 so it doesn't matter.  In case of a sequence l[2]=tokencnt,
# but we want the total loss, because that is what determined dw.
