using Base.Test
using CUDArt
using Knet

# import Knet: forw, back, ninputs, param, similar!, gpu, initforw, initback, push, pop, get1
# include("../src/net.jl")

include("isapprox.jl")


info("TEST 5")

isdefined(:MNIST) || include("mnist.jl")
net = [Drop(0.5), Conv(20,5), Bias(), Relu(), Pool(2),
       Drop(0.5), Conv(50,5), Bias(), Relu(), Pool(2),
       Drop(0.5), Mmul(500), Bias(), Relu(),
       Drop(0.5), Mmul(10), Bias(), XentLoss()]

nb = 128
x = KUdense(gpucopy(reshape(MNIST.xtrn[:,1:nb],28,28,1,nb)))
forw(net, copy(x))   # initializes the weights
rnn = Net(gpucopy(net)...)

setseed(42)
# @date y1 = forw(net, copy(x))
ybuf1 = Array(Any, length(net))
y = copy(x)
for n=1:length(net)
    y = forw(net[n], y)
    ybuf1[n] = cpucopy(y.arr)
end
y1 = y
    
setseed(42)
ybuf2 = Array(Any, length(net))
y = copy(x)
a = ()
r = rnn
inputs = Any[y]
trn = true
seq = false
using Knet: push, pop, dbg, forw, back, get1, initbatch

    initbatch(r, inputs...; trn=trn, seq=seq, a...)
    for i = 1:ninputs(r)
        n = i+nops(r)                           # input[i] goes into out[i+nops(r)]
        eltype(inputs[i]) == eltype(r.out0[n]) || error("Element type mismatch $i $n")
        trn && r.push[n] && push(r,n)         # t:140 save old input if necessary
        r.out[n] = copy!(r.out0[n], inputs[i]) 	# ; dbg(r,:out,n) # t:98 inputs can be any type of array, this will copy it to gpu or wherever
    end
    for n = 1:nops(r)
        trn && r.push[n] && push(r,n)         # t:327
        r.out[n] = forw(r.op[n], r.out[r.inputs[n]]...; y=r.out0[n], a...)     # ;dbg(r,:out,n) # t:2300
        ybuf2[n] = cpucopy(r.out[n].arr)
    end
    
y2 = r.out[nops(r)]

@show map(isequal, ybuf1, ybuf2)
@test @show to_host(y1.arr)==to_host(y2.arr)

y = KUdense(gpucopy(MNIST.ytrn[:,1:nb]))

# @date back(net, copy(y))
dy = copy(y)
dybuf1 = Array(Any, length(net))
for n=length(net):-1:1
    dy = back(net[n], dy)
    # @show (n, to_host(dy)[1])
    dybuf1[n] = cpucopy(dy.arr)
end
# @show length(dybuf1)

# @date back(rnn, copy(y))
r = rnn
dy = copy(y)
dx = nothing
seq = false
dybuf2 = Array(Any, nops(r))


    n = nops(r)
    if dy == nothing
        r.multi[n] || (r.dif[n] = nothing)
    elseif eltype(dy) != eltype(r.dif0[n])
        error("Element type mismatch $n")
    elseif r.multi[n]
        copy!(r.dif1[n], dy)
        r.dif[n] = axpy!(1,r.dif1[n],r.dif0[n])
    else
        r.dif[n] = copy!(r.dif0[n], dy)
    end										; dbg(r,:dif,n) 
    for n = nops(r):-1:1
        if r.dif[n] == nothing
            for i in r.inputs[n]
                r.multi[i] || (r.dif[i] = nothing)
            end
        else
            dxn = Any[]
            for i in r.inputs[n]
                push!(dxn, r.multi[i] ? r.dif1[i] : r.dif0[i])
            end
            back(r.op[n], r.dif[n]; incr=seq, x=get1(r.out[r.inputs[n]]), y=r.out[n], dx=get1(dxn), a...) # t:2164

### back gives eq for layers 18..8, approxeq for layers 7..1: TODO: investigate why
        dybuf2[n] = cpucopy(get1(dxn).arr)
        dytest1 = isapprox(dybuf1[n], dybuf2[n])
        @test dytest1
        dytest2 = isequal(dybuf1[n], dybuf2[n])
        p1 = map(p->p.diff, params(rnn.op[n]))
        p2 = map(p->p.diff, params(net[n]))
        dwtest1 = reduce(&, map(isapprox, p1, p2))
        @test dwtest1
        dwtest2 = reduce(&, map(isequal, p1, p2))
        println((n, typeof(rnn.op[n]), dytest1, dytest2, dwtest1, dwtest2))
###

            for i in r.inputs[n]
                r.multi[i] && axpy!(1, r.dif1[i], r.dif0[i])            ; r.multi[i]&&dbg(r,:dif1,i)
                r.dif[i] = r.dif0[i]                                    ; dbg(r,:dif,i)
            end
            r.multi[n] && fill!(r.dif[n],0)                           ; r.multi[n]&&dbg(r,:dif,n) # t:157
        end
        r.push[n] && pop(r,n)                                    ; r.push[n]&&dbg(r,:out,n)
    end
    for i = ninputs(r):-1:1
        n = i+nops(r)
        r.push[n] && pop(r,n)                                    ; r.push[n] && dbg(r,:out,n)
        dx == nothing || copy!(dx[i], r.dif[n])
    end



# # initback(r)
# n = nops(r)


# r.dif[n] = copy!(r.dif0[n], dy)             # println("back:dy=$((idx1(getdy(r,nops(r))),))")    
# for n = nops(r):-1:1
#     if r.dif[n] != nothing                  # 'nothing' represents 0 loss gradient
#         dx = map(r.inputs[n]) do i; r.inc[i]!=nothing ? r.inc[i] : (r.dif[i]=r.dif0[i]); end
#         dy = back(r.op[n], r.dif[n]; incr=true, x=get1(r.out[r.inputs[n]]), y=r.out[n], dx=get1(dx), a...)

# ### back gives eq for layers 18..8, approxeq for layers 7..1
#         dybuf2[n] = cpucopy(dy.arr)
#         dytest1 = isapprox(dybuf1[n], dybuf2[n])
#         @test dytest1
#         dytest2 = (dybuf1[n] == dybuf2[n])
#         p1 = map(p->p.diff, params(rnn.op[n]))
#         p2 = map(p->p.diff, params(net[n]))
#         @test dwtest1 = all(map(isapprox, p1, p2))
#         dwtest2 = all(map(isequal, p1, p2))
#         println((n, typeof(rnn.op[n]), dytest1, dytest2, dwtest1, dwtest2))
# ###

#         for i in r.inputs[n]; r.inc[i]!=nothing && axpy!(1, r.inc[i], r.dif[i]); end
#         r.inc[n]!=nothing && fill!(r.dif[n],0)
#     end                                     # println("op[$n]:$((typeof(r.op[n]),:x,map(idx1,getx(r,n))...,:y,idx1(gety(r,n)),:dy,idx1(getdy(r,n)),:dx,map(idx1,getdxbuf(r,n))...)) st=$(map(idx1,r.stack[1:r.sp]))")
#     r.push[n] && pop(r,n)                   # println("pop[$n]:y=$((idx1(gety(r,n)),)) st=$(map(idx1,r.stack[1:r.sp]))")
# end
# for i = ninputs(r):-1:1                     # println("in[$i]=$(map(idx1,(getinput(r,i),))) st=$(map(idx1,r.stack[1:r.sp]))")
#     n = i+nops(r)                           # println("in[$i]=$(map(idx1,(getinput(r,i),))) st=$(map(idx1,r.stack[1:r.sp]))")
#     r.push[n] && pop(r,n)
# end


if false

info("TEST 4")

isdefined(:MNIST) || include("mnist.jl")
setseed(42)
net = [Conv(20,5), Bias(), Relu(), Pool(2),
       Conv(50,5), Bias(), Relu(), Pool(2),
       Mmul(500), Bias(), Relu(),
       Mmul(10), Bias(), XentLoss()]

nb = 128
x = KUdense(gpucopy(reshape(MNIST.xtrn[:,1:nb],28,28,1,nb)))
y = KUdense(gpucopy(MNIST.ytrn[:,1:nb]))
@date y1 = forw(net, copy(x))   # also initializes the weights
rnn = Net(deepcopy(net)...)
init(rnn, copy(x); train=true)
@date y2 = forw(rnn, copy(x); y=similar(y1))
@test @show to_host(y1.arr)==to_host(y2.arr)

@date back(net, copy(y))
@date back(rnn, copy(y))
for i=nops(rnn):-1:1
    isempty(params(net[i])) && continue
    p1 = map(p->p.diff, params(rnn.op[i]))
    p2 = map(p->p.diff, params(net[i]))
    print("$i "); @test @show all(map(isapprox, p1, p2))
    print("$i "); @show all(map(isequal, p1, p2)) # 1,2,5 only approx
end

info("TEST 3")

x = KUdense(gpucopy(rand(784,10)))
net = Op[Mmul(10),QuadLoss()]
@date y1 = forw(net, copy(x))
rnn = Net(gpucopy(net)...)
init(rnn, copy(x))
@date y2 = forw(rnn, copy(x); y=similar(y1))
@test @show to_host(y1.arr)==to_host(y2.arr)
dy = rand!(copy(y1))
back(net, copy(dy))
back(rnn, copy(dy))
for i=1:nops(rnn)
    isempty(params(net[i])) && continue
    p1 = map(p->p.diff, params(rnn.op[i]))
    p2 = map(p->p.diff, params(net[i]))
    print("$i "); @test @show all(map(isequal, p1, p2)) # 1,2,5 only approx
end

info("TEST 2")

isdefined(:MNIST) || include("mnist.jl")
setseed(42)
nb = 100
x = KUdense(gpucopy(MNIST.xtrn[:,1:nb]))
net = [Mmul(64), Bias(), Relu(), 
       Mmul(10), Bias(), XentLoss()]
@date y1 = forw(net, copy(x))

rnn = Net(gpucopy(net)...)
init(rnn, copy(x))
@date y2 = forw(rnn, copy(x); y=similar(y1))

# @show isapprox(y1,y2)
@test @show to_host(y1.arr)==to_host(y2.arr)

y = KUdense(gpucopy(MNIST.ytrn[:,1:nb]))
back(net, copy(y))
back(rnn, copy(y))
for i=1:nops(rnn)
    isempty(params(net[i])) && continue
    p1 = map(p->p.diff, params(rnn.op[i]))
    p2 = map(p->p.diff, params(net[i]))
    print("$i "); @test @show all(map(isequal, p1, p2)) # 1,2,5 only approx
end

info("TEST 1")
# irnn(h)=Net(Mmul(h), (Mmul(h),5), Add2(), Bias(), Relu())
# add1(h)=Net(Mmul(h), (Mmul(h),-1), Add2(), Bias(), Sigm())
# add2(h)=Net(Mmul(h), (Mmul(h),-1), Add2(), Bias(), Tanh())
# lstm(h)=Net((add1(h),0,9),      # 1. i
#             (add1(h),0,9),      # 2. f
#             (add1(h),0,9),      # 3. o
#             (add2(h),0,9),      # 4. cc
#             (Mul2(),1,4),       # 5. i*cc
#             (Mul2(),2,7),       # 6. f*c[t-1]
#             Add2(),             # 7. c
#             Tanh(),             # 8. tanh(c)
#             (Mul2(),3,8))       # 9. o * tanh(c)

function testops(a,ops)
    nops(a) == length(ops) || return false
    for i=1:nops(a)
        isa(a.op[i], ops[i]) || return false
    end
    return true
end

a = irnn(10)
aops = [Mmul, Mmul, Add2, Bias, Relu]
@test @show testops(a, aops)
@test @show a.inputs == Any[[6],[5],[1,2],[3],[4]]
@test @show a.ninputs == 1
@test @show find(a.push) == [5,6]
#@test @show a.y == [1,2,1,1,1,3]
#@test @show a.dy == [5,4,5,5,6,7]
#@test @show all(a.out .== nothing)
#@test @show all(a.out0 .== nothing)
@test @show a.stack == Any[]
@test @show a.sp == 0

b = Net(irnn(10),irnn(10))
bops = vcat(aops,aops)
@test @show testops(b, bops)
@test @show b.inputs == Any[[11],[5],[1,2],[3],[4],[5],[10],[6,7],[8],[9]]
@test @show b.ninputs == 1
@test @show find(b.push) == [5,10,11]
#@test @show b.y == [1,2,1,1,1,3,4,3,3,3,5]
#@test @show b.dy == [7,6,7,7,8,10,9,10,10,11,12]
#@test @show all(b.out .== nothing)
#@test @show b.out0 == b.out
@test @show b.stack == Any[]
@test @show b.sp == 0

c = lstm(10)
cops = [Mmul,Mmul,Add2,Bias,Sigm,Mmul,Mmul,Add2,Bias,Sigm,Mmul,Mmul,Add2,Bias,Sigm,Mmul,Mmul,Add2,Bias,Tanh,Mul2,Mul2,Add2,Tanh,Mul2]
@test @show testops(c, cops)
@test @show c.inputs == Any[[26],[25],[1,2],[3],[4],[26],[25],[6,7],[8],[9],[26],[25],[11,12],[13],[14],[26],[25],[16,17],[18],[19],[5,20],[10,23],[21,22],[23],[15,24]]
@test @show c.ninputs == 1
@test @show find(c.push) == [5,10,15,20,23,24,25,26]
#@test @show c.y == [1,2,1,1,1,3,4,3,3,3,5,6,5,5,5,7,8,7,7,7,9,10,9,11,12,13]
#@test @show c.dy == [15,14,15,15,15,17,16,17,17,17,19,18,19,19,19,21,20,21,21,21,22,23,24,25,26,27]
#@test @show all(c.out .== nothing)
#@test @show c.out0 == c.out
@test @show c.stack == Any[]
@test @show c.sp == 0



end # if false

:ok