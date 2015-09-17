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
        $layer(n)=Net(Mmul(n), Bias(), $opname())
        export $layer
    end
end

add2(n)=Net(Mmul(n), (Mmul(n),-1), Add2(), Bias())

lstm(n)=Net((add2(n),0,13), Sigm(),     # 1-2. input
            (add2(n),0,13), Sigm(),     # 3-4. forget
            (add2(n),0,13), Sigm(),     # 5-6. output
            (add2(n),0,13), Tanh(),     # 7-8. cc
            (Mul2(),2,8), (Mul2(),4,11), Add2(), # 9-11. c = i*cc + f*c[t-1]
            Tanh(), (Mul2(),6,12))      # 12-13. h = o * tanh(c)

eye!(a)=copy!(a, eye(eltype(a), size(a)...)) # TODO: don't alloc

irnn(n)=Net(Mmul(n; init=randn!, initp=(0,0.001)), 
            (Mmul(n; init=eye!), 5), 
            Add2(), Bias(), Relu())

# S2C: sequence to class model

immutable S2C <: Model; net1; net2; params;
    S2C(a,b)=new(a,b,vcat(params(a),params(b)))
end

params(r::S2C)=r.params

loss(r::S2C,y)=loss(r.net2,y)

function forw(r::S2C, x::Vector; y=nothing, a...)
    n = nops(r.net1)            
    forw(r.net1, x; a...)
    forw(r.net2, r.net1.out[n]; y=y, a...)   # passing a sequence [n:n] to net2 forces init
    # TODO: implement lastout or direct write from net1 out to net2 input
    # You can do the latter if net2 is initialized first or just overwrite its buf0
    # and check for === before copy
end

function back(r::S2C, y)
    back(r.net2, y)
    back(r.net1, r.net2.dif[nops(r.net2)+1]; seq=true)
    while r.net1.sp > 0; back(r.net1, nothing; seq=true); end
end

