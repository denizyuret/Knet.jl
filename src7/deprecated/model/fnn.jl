immutable FNN <: Model; net; FNN(a...;o...)=new(Net(a...;o...)); end
params(m::FNN)=params(m.net)

function train(m::FNN, data, loss; gclip=0, gcheck=false, maxnorm=nothing, losscnt=nothing, o...)
    for item in data
        (x,ygold) = item2xy(item)
        reset!(m.net; o...)
        ypred = forw(m.net, x...; trn=true, o...)
        losscnt != nothing && (losscnt[1] += loss(ypred, ygold); losscnt[2] += 1)
        back(m.net, ygold, loss; o...)
        gcheck && break
        g = (gclip > 0 || maxnorm!=nothing ? gnorm(m) : 0)
        update!(m; gclip=(g > gclip > 0 ? gclip/g : 0), o...)
        if maxnorm != nothing
            w=wnorm(m)
            w > maxnorm[1] && (maxnorm[1]=w)
            g > maxnorm[2] && (maxnorm[2]=g)
        end
    end
end

function test(m::FNN, data, loss; gcheck=false, o...)
    sumloss = numloss = 0
    for item in data
        (x,ygold) = item2xy(item)
        ypred = forw(m.net, x...; trn=false, o...)
        sumloss += loss(ypred, ygold); numloss += 1
        gcheck && break
    end
    sumloss / numloss
end

function predict(m::FNN, data; o...)
    y = Any[]
    for x in data
        ypred = forw(m.net, x...; trn=false, o...)
        ycopy = isa(ypred, Array) ? copy(ypred) : convert(Array, ypred)
        push!(y, ycopy)
    end
    return y
end

# function gradcheck(m::FNN, data, loss; delta=1e-4, rtol=eps(Float64)^(1/5), atol=eps(Float64)^(1/5), gcheck=10, o...)
#     x = ygold = loss0 = nothing
#     for item in data
#         (x,ygold) = item2xy(item)
#         ypred = forw(m.net, x...; trn=true, o...)
#         loss0 = loss(ypred, ygold)
#         back(m.net, ygold, loss; o...)
#         break
#     end
#     pp = params(m)
#     for n=1:length(pp)
#         p = pp[n]
#         psave = copy(p.out)
#         pdiff = convert(Array, p.dif)
#         wlen = length(p.out)
#         irange = (wlen <= gcheck ? (1:wlen) : rand(gradcheck_rng, 1:wlen, gcheck))
#         for i in irange
#             wi0 = p.out[i]
#             wi1 = (wi0 >= 0 ? wi0 + delta : wi0 - delta)
#             p.out[i] = wi1
#             ypred = forw(m.net, x...; trn=false, o...)
#             loss1 = loss(ypred, ygold)
#             p.out[i] = wi0
#             dwi = (loss1 - loss0) / (wi1 - wi0)
#             if !isapprox(pdiff[i], dwi; rtol=rtol, atol=atol)
#                 println(tuple(:gc, n, i, pdiff[i], dwi))
#             end
#         end
#         @assert isequal(p.out, psave)
#     end
# end

