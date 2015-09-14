for (layer, opname) in 
    ((:sigmlayer, :Sigm),
     (:tanhlayer, :Tanh),
     (:relulayer, :Relu),
     (:softlayer, :Soft),
     (:logplayer, :Logp),
     (:quadlosslayer, :QuadLoss),
     (:softlosslayer, :SoftLoss),
     (:logplosslayer, :LogpLoss),
     (:xentlosslayer, :XentLoss),
     (:perclosslayer, :PercLoss),
     (:scallosslayer, :ScalLoss),
     )
    @eval begin
        $layer(n)=RNN(Mmul(n), Bias(), $opname())
        export $layer
    end
end

add2(n)=RNN(Mmul(n), (Mmul(n),-1), Add2(), Bias())

lstm(n)=RNN((add2(n),0,13), Sigm(),     # 1-2. input
            (add2(n),0,13), Sigm(),     # 3-4. forget
            (add2(n),0,13), Sigm(),     # 5-6. output
            (add2(n),0,13), Tanh(),     # 7-8. cc
            (Mul2(),2,8), (Mul2(),4,11), Add2(), # 9-11. c = i*cc + f*c[t-1]
            Tanh(), (Mul2(),6,12))      # 12-13. h = o * tanh(c)

eye!(a)=copy!(a, eye(eltype(a), size(a)...)) # TODO: don't alloc

irnn(n)=RNN(Mmul(n; init=randn!, initp=(0,0.001)), 
            (Mmul(n; init=eye!), 5), 
            Add2(), Bias(), Relu())

type S2C <: Model; net1; net2; nt; S2C(a,b)=new(a,b,0); end

function forw(r::S2C, x::Vector; y=nothing, a...)
    n = nops(r.net1)
    r.nt = length(x)
    forw(r.net1, x; a...)
    forw(r.net2, r.net1.out[n:n]; y=y, a...)   # passing a sequence [n:n] to net2 forces init
end

function back(r::S2C, y)
    back(r.net2, y)
    back(r.net1, r.net2.dif[nops(r.net2)+1])
    for t=1:r.nt-1; back(r.net1, nothing); end
end

loss(r::S2C,y)=loss(r.net2,y)
update(r::S2C)=(update(r.net1); update(r.net2))
nops(r::S2C)=nops(r.net1)+nops(r.net2)
op(r::S2C,n)=(n1=nops(r.net1); n<=n1 ? r.net1.op[n] : r.net2.op[n-n1])
Base.isequal(a::S2C,b::S2C)=isequal(a.net1,b.net1)&&isequal(a.net2,b.net2)

### DEAD CODE:

# DONE: create a super class for RNN and S2C, this gradcheck applies to both
# think about the common interface for ops, nets, and other arch models.

# function forwback(r::S2C,x,y; getloss=false)
#     forw(r, x; train=true)
#     getloss && (l0 = loss(r, y))
#     back(r, y)
#     getloss && return l0
# end

# function train(r::S2C,x,y; getloss=false, getnorm=false, gc=0)
#     rval = Any[]
#     l = forwback(r,x,y; getloss=getloss)
#     (getnorm || gc>0) && ((w,g)=sumnorm(r))
#     update(r)
#     g > gc > 0 && gscale!(r, gc/g)
#     getloss && push!(rval, l)
#     getnorm && push!(rval, w, g)
#     return tuple(rval...)
# end

# function loss(r::S2C, x, y)
#     forw(r, x; train=false)
#     loss(r, y)
# end

# function gradcheck(r::S2C, x, y; delta=1e-4, rtol=eps(Float64)^(1/5), atol=eps(Float64)^(1/5), ncheck=10)
#     l0 = forwback(r, x, y; getloss=true)
#     dw = cell(nops(r))
#     for n=1:length(dw)
#         p = param(op(r,n))
#         dw[n] = (p == nothing ? nothing : convert(Array, p.diff))
#     end
#     for n=1:length(dw)
#         dw[n] == nothing && continue
#         p = param(op(r,n))
#         w = convert(Array, p.arr)
#         wlen = length(w)
#         irange = (wlen <= ncheck ? (1:wlen) : rand(1:wlen, ncheck))
#         for i in irange
#             wi0 = w[i]
#             wi1 = (wi0 >= 0 ? wi0 + delta : wi0 - delta)
#             w[i] = wi1
#             copy!(p.arr, w)
#             l1 = loss(r,x,y)
#             w[i] = wi0 
#             dwi = (l1 - l0) / (wi1 - wi0)
#             if !isapprox(dw[n][i], dwi; rtol=rtol, atol=atol)
#                 println(tuple(:gc, n, i, dw[n][i], dwi))
#             end
#         end
#         copy!(p.arr, w)         # make sure we recover the original
#     end
# end

# # Same goes here
# function maxnorm(r::S2C, mw=0, mg=0)
#     for n=1:nops(r)
#         p = param(op(r,n))
#         p == nothing && continue
#         w = vecnorm(p.arr)
#         w > mw && (mw = w)
#         g = vecnorm(p.diff)
#         g > mg && (mg = g)
#     end
#     return (mw, mg)
# end

# function sumnorm(r::S2C, w=0, g=0)
#     for n=1:nops(r)
#         p = param(op(r,n))
#         p == nothing && continue
#         w += vecnorm(p.arr)
#         g += vecnorm(p.diff)
#     end
#     return (w, g)
# end

# function gscale!(r::S2C, s)
#     for n=1:nops(r)
#         p = param(op(r,n))
#         p == nothing && continue
#         scale!(s, p.diff)
#     end
# end
