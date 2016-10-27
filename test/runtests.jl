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
#Unpooling
x = KnetArray(reshape(Float32[1.0:16.0...], (4,4,1,1)))
y = pool(x)
y2 = KnetArray(reshape(Float32[6 6 14 14; 6 6 14 14; 8 8 16 16; 8 8 16 16], (4,4,1,1)))
y3 = KnetArray(reshape(Float32[6 6 6 14 14 14; 6 6 6 14 14 14; 6 6 6 14 14 14; 8 8 8 16 16 16; 8 8 8 16 16 16; 8 8 8 16 16 16], (6,6,1,1)))

@test_approx_eq_eps     myunpool(y)             y2      1e-3
@test_approx_eq_eps     myunpool(y; window=3)   y3      1e-3

#Deconvolution
y = KnetArray(reshape(Float32[0 10 20 30; 20 110 170 150; 80 290 350 270; 140 370 420 270], (4,4,1,1)))
x = KnetArray(reshape(Float32[0.0 10.0; 20.0 30.0], (2,2,1,1)))
w = KnetArray(reshape(Float32[1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0], (3,3,1,1)))

@test_approx_eq_eps     deconv4(w,x)            y       1e-3