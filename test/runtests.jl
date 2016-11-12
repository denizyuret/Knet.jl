# This takes too long on Travis:
# Pkg.add("ArgParse")
# load_only=true
# include(Knet.dir("examples","charlm.jl"))
# CharLM.main("--gcheck 3")

using Knet
srand(42)

predict(w,x)=(w*x)

loss(w,x,y)=(sum((y-predict(w,x)).^2) / size(x,2))

lossgradient = grad(loss)

function train(w, data; lr=.02, epochs=10)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(w, x, y)
            w -= lr * g
        end
    end
    return w
end

function test(w, data)
    sumloss = numloss = 0
    for (x,y) in data
        sumloss += loss(w,x,y)
        numloss += 1
    end
    return sumloss/numloss
end

# Data generator:
import Base: start, next, done

type Data; w; batchsize; epochsize; noise; rng; atype; end

function Data(outputdims,inputdims; batchsize=20, epochsize=10000, noise=.01, rng=Base.GLOBAL_RNG, atype=Array)
    Data(convert(atype, randn(rng,outputdims,inputdims)),batchsize,epochsize,noise,rng,atype)
end

function next(l::Data, n)
    (outputdims, inputdims) = size(l.w)
    x = convert(l.atype, rand(l.rng, inputdims, l.batchsize))
    y = l.w * x + convert(l.atype, l.noise * randn(l.rng, outputdims, l.batchsize))
    return ((x,y), n+l.batchsize)
end

start(l::Data)=0
done(l::Data,n)=(n >= l.epochsize)


# Run:

data = Data(10,100)
w = 0.1*randn(10,100)
println((:epoch,0,:loss,test(w,data)))
@time for epoch=1:10
    w = train(w, data; epochs=1, lr=0.02)
    println((:epoch,epoch,:loss,test(w,data)))
end


#Tests for features I implemented
using Base.Test;
function isapprox4{T}(calculated::KnetArray{T},expected::KnetArray{T},c=1e-3)
    calculated = Array(calculated)
    expected = Array(expected)
    c = reshape(ones(T,length(expected))*c,size(expected))
    all(map((x,y,z)->isapprox(x,y;rtol=z), calculated,expected,c))
end

#Unpooling
info("Testing unpooling (forward pass)...")
x1 = KnetArray(reshape(Float32[ 6 14;
                                8 16], (2,2,1,1)))
x2 = KnetArray(reshape(Float32[1.0:9.0...], (3,3,1,1)))
y12 = KnetArray(reshape(Float32[6 6 14 14;
                                6 6 14 14;
                                8 8 16 16;
                                8 8 16 16], (4,4,1,1)))
y13 = KnetArray(reshape(Float32[6 6 6 14 14 14;
                                6 6 6 14 14 14;
                                6 6 6 14 14 14;
                                8 8 8 16 16 16;
                                8 8 8 16 16 16;
                                8 8 8 16 16 16], (6,6,1,1)))
y22 = KnetArray(reshape(Float32[1 1 4 4 7 7;                                                                                                                                           
                                1 1 4 4 7 7;                                                                                                                                          
                                2 2 5 5 8 8;                                                                                                                                         
                                2 2 5 5 8 8;                                                                                                                                           
                                3 3 6 6 9 9;                                                                                                                                           
                                3 3 6 6 9 9], (6,6,1,1)))
y23 = KnetArray(reshape(Float32[1 1 1 4 4 4 7 7 7;                                                                                                                            
                                1 1 1 4 4 4 7 7 7;                                                                                                                           
                                1 1 1 4 4 4 7 7 7;                                                                                                                           
                                2 2 2 5 5 5 8 8 8;                                                                                                                           
                                2 2 2 5 5 5 8 8 8;                                                                                                                           
                                2 2 2 5 5 5 8 8 8;                                                                                                                           
                                3 3 3 6 6 6 9 9 9;                                                                                                                           
                                3 3 3 6 6 6 9 9 9;                                                                                                                            
                                3 3 3 6 6 6 9 9 9], (9,9,1,1)))
#Even input, even and odd windows
@test isapprox4(unpool(x1),y12)
@test isapprox4(unpool(x1; window=3),y13)
#Odd input, even and odd windows
@test isapprox4(unpool(x2), y22)
@test isapprox4(unpool(x2; window=3),y23)
#All possible cases covered





info("Testing unpooling (backward pass)...")
dy1 = KnetArray(reshape(Float32[1 1 0 0;
                                1 1 0 0;
                                0 0 0 0;
                                0 0 0 0], (4,4,1,1)))
dy2 = KnetArray(reshape(Float32[0 0 1 1;
                                0 0 1 1;
                                0 0 0 0;
                                0 0 0 0], (4,4,1,1)))
dy3 = KnetArray(reshape(Float32[0 0 0 0;
                                0 0 0 0;
                                1 1 0 0;
                                1 1 0 0], (4,4,1,1)))
dy4 = KnetArray(reshape(Float32[0 0 0 0;
                                0 0 0 0;
                                0 0 1 1;
                                0 0 1 1], (4,4,1,1)))

#Gradients should be pool(dy) for the following cases
@test isapprox4(Knet.unpoolx(dy1), pool(dy1))
@test isapprox4(Knet.unpoolx(dy2), pool(dy2))
@test isapprox4(Knet.unpoolx(dy3), pool(dy3))
@test isapprox4(Knet.unpoolx(dy4), pool(dy4))

dyW = KnetArray(reshape(Float32[1 0 0 0;
                                1 0 0 0;
                                0 4 0 0;
                                1 2 0 0], (4,4,1,1)))
xW = KnetArray(reshape(Float32[ 0  0;
                                0  0], (2,2,1,1)))

#Gradient should be zero for the following case, since the output
#cant't change like this for x under any condition
@test isapprox4(Knet.unpoolx(dyW), xW)
#All possible cases covered













#Deconvolution
info("Testing deconvolution (forward pass)...")
y = KnetArray(reshape(Float32[0 10 20 30; 20 110 170 150; 80 290 350 270; 140 370 420 270], (4,4,1,1)))
x = KnetArray(reshape(Float32[0.0 10.0; 20.0 30.0], (2,2,1,1)))
w = KnetArray(reshape(Float32[1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0], (3,3,1,1)))

@test isapprox4(deconv4(w,x),y)

info("Testing deconvolution (backward pass)...")

#=
info("Testing Float16...")

#GPU Float64
w = KnetArray(reshape(Float64[1.0 2.0 3.0 4.0], (2,2)))
x = KnetArray(reshape(Float64[1.0 1.0 1.0 1.0], (2,2)))
y = KnetArray(Float64[4.0 4.0; 6.0 6.0])
@test isapprox4(w*x, y)

#GPU Float32
w = KnetArray(reshape(Float32[1.0 2.0 3.0 4.0], (2,2)))
x = KnetArray(reshape(Float32[1.0 1.0 1.0 1.0], (2,2)))
y = KnetArray(Float32[4.0 4.0; 6.0 6.0])
@test isapprox4(w*x, y) 

#GPU Float16
w = KnetArray(reshape(Float16[1.0 2.0 3.0 4.0], (2,2)))
x = KnetArray(reshape(Float16[1.0 1.0 1.0 1.0], (2,2)))
y = KnetArray(Float16[4.0 4.0; 6.0 6.0])
@test isapprox4(w*x, y)
=#