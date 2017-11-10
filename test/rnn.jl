using Knet
include(Knet.dir("src/rnn.jl"))  # will be removed after integration

dt = Float32
(r,w) = rnninit(64,100; dataType=dt)
ka = KnetArray
x1 = ka(randn(dt,64))
x2 = ka(randn(dt,64,16))
x3 = ka(randn(dt,64,16,10))
foo(w,x,r)=sum(rnn(r,w,x)[1])
@show gradcheck(foo,w,x1,r; verbose=true,rtol=0.1)
@show gradcheck(foo,w,x2,r; verbose=true,rtol=0.1)
@show gradcheck(foo,w,x3,r; verbose=true,rtol=0.1)
