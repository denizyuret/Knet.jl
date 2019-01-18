using Knet
using Statistics: mean
using Base.Iterators: take

# Define convolutional layer:
struct Conv; w; b; f; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
Conv(w1,w2,cx,cy,f=relu) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f)

# Define dense layer:
struct Dense; w; b; f; end
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)
Dense(i::Int,o::Int,f=relu) = Dense(param(o,i), param0(o), f)

# Define a chain of layers:
struct Chain; layers; Chain(args...) = new(args); end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)
(c::Chain)(d::Array{Any,1}) = mean(c(x,y) for (x,y) in d)

# Load MNIST data
include(Knet.dir("data","mnist.jl"))
dtrn, dtst = mnistdata()

# Train and test LeNet (about 30 secs on a gpu to reach 99% accuracy)
lenet() = Chain(Conv(5,5,1,20), Conv(5,5,20,50), Dense(800,500), Dense(500,10,identity))

sgdopt(x;ep=1) = (Knet.seed!(1); m=lenet(); sgd!(m,repeat(dtrn,ep),lr=10^(x[1]-1)); m(dtrn))
# goldensection(sgdopt, 1, verbose=true, dxmin=0.01)
# sgd-lenet1: (0.06583427f0, [0.493422])
# sgd-lenet2: (0.056056067f0, [0.673762])
# sgd-lenet3: (0.056493137f0, [0.673762])
momopt(x;ep=1) = (Knet.seed!(1); m=lenet(); momentum!(m,repeat(dtrn,ep),lr=10^(x[1]-3),gamma=1-10^(x[2]-1)); m(dtrn))
# goldensection(momopt, 2, verbose=true, dxmin=0.01)
# mom-lenet1: (0.04570869f0, [1.38197, -0.416408])
# mom-lenet2: (0.044232562f0, [1.36068, -0.416408])
# mom-lenet3: (0.04657365f0, [1.38197, -0.381966])
# mom-lenet4: (0.045377113f0, [1.52786, -0.326238])
# mom-lenet5: (0.045804564f0, [1.38197, -0.291796])
# mom-lenet6: (0.04432534f0, [1.38197, -0.291796])
nesopt(x;ep=1) = (Knet.seed!(1); m=lenet(); momentum!(m,repeat(dtrn,ep),lr=10^(x[1]-3),gamma=1-10^(x[2]-1)); m(dtrn))
# goldensection(nesopt, 2, verbose=true, dxmin=0.01)
# nes-lenet1: (0.044091783, [1.38197, -0.416408])
# nes-lenet2: (0.046030354, [1.38197, -0.291796])
# nes-lenet3: (0.044058617, [1.38197, -0.291796])
adgopt(x;ep=1) = (Knet.seed!(1); m=lenet(); adagrad!(m,repeat(dtrn,ep),lr=10^(x[1]-1),eps=10^(x[2]-6)); m(dtrn))
# [goldensection(adgopt, 2, verbose=true, dxmin=0.01) for i in 1:3]
# (0.04346694, [-0.763932, 0.159054])  
# (0.045540974, [-0.742646, -0.618034])
# (0.049198333, [-0.381966, 1.0])      
addopt(x;ep=1) = (Knet.seed!(1); m=lenet(); adadelta!(m,repeat(dtrn,ep),lr=10^(x[1]-2),rho=1-10^(x[2]-1),eps=10^(x[3]-6)); m(dtrn))
# [goldensection(addopt, 3, verbose=true, dxmin=0.01) for i in 1:3]
 # (0.058205813, [1.56231, -0.63932, 2.60488])
 # (0.043803316, [2.61803, -0.618034, -1.0])  
 # (0.05949348, [1.38197, 0.381966, 5.21478]) 
rmsopt(x;ep=2) = (Knet.seed!(1); m=lenet(); rmsprop!(m,repeat(dtrn,ep),lr=10^(x[1]-2),rho=1-10^(x[2]-1),eps=10^(x[3]-6)); m(dtrn))
# [goldensection(rmsopt, 3, verbose=true, dxmin=0.01) for i in 1:3]
 # (0.027730716, [-0.618034, -1.23607, 0.0])        
 # (0.024157057, [-0.618034, -0.0901699, -0.304952])
 # (0.025530454, [-0.583592, -0.618034, 0.0])       

admopt(x;ep=1) = (Knet.seed!(1); m=lenet(); adam!(m,repeat(dtrn,ep),lr=10^(x[1]-3),beta1=1-10^(x[2]-1),beta2=1-10^(x[3]-3),eps=10^(x[4]-8)); m(dtrn))
# lenet1=(0.03586271f0, [0.562306, 0.0, -1.23607, 0.0])
# lenet2=(0.035605736f0, [0.618034, -0.201626, 0.0, 2.61803])
 #         (0.03902738, [0.506578, -0.583592, 0.0212862, 0.381966]) 
 #         (0.036795303, [0.381966, -0.257354, -0.618034, 0.673762])
 #        (0.041281685, [0.381966, -0.604878, 0.0901699, 0.145898])

# Load Brown data
include(Knet.dir("data/nltk.jl"))
(data,words,tags) = brown()

struct Dense2; w; b; f; end
Dense2(i::Int,o::Int,f=identity) = Dense2(param(o,i), param0(o), f)
reshape2d(x) = reshape(x,(size(x,1),:))
(d::Dense2)(x) = d.f.(d.w * reshape2d(x) .+ d.b)

struct Embed; w; end
Embed(vocabsize::Int,embedsize::Int) = Embed(param(embedsize,vocabsize))
(e::Embed)(x) = e.w[:,x]

BATCHSIZE = 64
SEQLENGTH = 32;
VOCABSIZE = length(words)
EMBEDSIZE = 128
HIDDENSIZE = 128
OUTPUTSIZE = length(tags);

Tagger0(vocab=VOCABSIZE,embed=EMBEDSIZE,hidden=HIDDENSIZE,output=OUTPUTSIZE)=  # MLP Tagger
    Chain(Embed(vocab,embed),Dense2(embed,hidden,relu),Dense2(hidden,output,identity))
Tagger1(vocab=VOCABSIZE,embed=EMBEDSIZE,hidden=HIDDENSIZE,output=OUTPUTSIZE)=  # RNN Tagger
    Chain(Embed(vocab,embed),RNN(embed,hidden,rnnType=:relu),Dense2(hidden,output,identity))
Tagger2(vocab=VOCABSIZE,embed=EMBEDSIZE,hidden=HIDDENSIZE,output=OUTPUTSIZE)=  # biRNN Tagger
    Chain(Embed(vocab,embed),RNN(embed,hidden,rnnType=:relu,bidirectional=true),Dense2(2hidden,output,identity));

function seqbatch(x,y,B,T)
    N = length(x) ÷ B
    x = permutedims(reshape(x[1:N*B],N,B))
    y = permutedims(reshape(y[1:N*B],N,B))
    d = []; for i in 0:T:N-T
        push!(d, (x[:,i+1:i+T], y[:,i+1:i+T]))
    end
    return d
end
allw = vcat((x->x[1]).(data)...)
allt = vcat((x->x[2]).(data)...)
dtag = seqbatch(allw, allt, BATCHSIZE, SEQLENGTH);

# t0 = Tagger0(VOCABSIZE,EMBEDSIZE,HIDDENSIZE,OUTPUTSIZE)
# t1 = Tagger1(VOCABSIZE,EMBEDSIZE,HIDDENSIZE,OUTPUTSIZE)
# t2 = Tagger2(VOCABSIZE,EMBEDSIZE,HIDDENSIZE,OUTPUTSIZE)

sgdoptz(x;ep=1) = (Knet.seed!(1); m=lenet();   sgd!(m,repeat(dtrn,ep),lr=10^(x[1]-1.0)); m(dtrn))
sgdopt0(x;ep=1) = (Knet.seed!(1); m=Tagger0(); sgd!(m,repeat(dtag,ep),lr=10^(x[1]-1.0)); m(dtag))
sgdopt1(x;ep=1) = (Knet.seed!(1); m=Tagger1(); sgd!(m,repeat(dtag,ep),lr=10^(x[1]-1.0)); m(dtag))
sgdopt2(x;ep=1) = (Knet.seed!(1); m=Tagger2(); sgd!(m,repeat(dtag,ep),lr=10^(x[1]-1.0)); m(dtag))

momoptz(x;ep=1) = (Knet.seed!(1); m=lenet();   momentum!(m,repeat(dtrn,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtrn))
momopt0(x;ep=1) = (Knet.seed!(1); m=Tagger0(); momentum!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))
momopt1(x;ep=1) = (Knet.seed!(1); m=Tagger1(); momentum!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))
momopt2(x;ep=1) = (Knet.seed!(1); m=Tagger2(); momentum!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))

nesoptz(x;ep=1) = (Knet.seed!(1); m=lenet();   nesterov!(m,repeat(dtrn,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtrn))
nesopt0(x;ep=1) = (Knet.seed!(1); m=Tagger0(); nesterov!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))
nesopt1(x;ep=1) = (Knet.seed!(1); m=Tagger1(); nesterov!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))
nesopt2(x;ep=1) = (Knet.seed!(1); m=Tagger2(); nesterov!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))

adgoptz(x;ep=1) = (Knet.seed!(1); m=lenet();   adagrad!(m,repeat(dtrn,ep),lr=10^(x[1]-1.75),eps=10^(x[2]-6)); m(dtrn))
adgopt0(x;ep=1) = (Knet.seed!(1); m=Tagger0(); adagrad!(m,repeat(dtag,ep),lr=10^(x[1]-1.75),eps=10^(x[2]-6)); m(dtag))
adgopt1(x;ep=1) = (Knet.seed!(1); m=Tagger1(); adagrad!(m,repeat(dtag,ep),lr=10^(x[1]-1.75),eps=10^(x[2]-6)); m(dtag))
adgopt2(x;ep=1) = (Knet.seed!(1); m=Tagger2(); adagrad!(m,repeat(dtag,ep),lr=10^(x[1]-1.75),eps=10^(x[2]-6)); m(dtag))

addoptz(x;ep=1) = (Knet.seed!(1); m=lenet();   adadelta!(m,repeat(dtrn,ep),lr=10^(x[1]),rho=1-10^(x[2]-1.6),eps=10^(x[3]-7)); m(dtrn))
addopt0(x;ep=1) = (Knet.seed!(1); m=Tagger0(); adadelta!(m,repeat(dtag,ep),lr=10^(x[1]),rho=1-10^(x[2]-1.6),eps=10^(x[3]-7)); m(dtag))
addopt1(x;ep=1) = (Knet.seed!(1); m=Tagger1(); adadelta!(m,repeat(dtag,ep),lr=10^(x[1]),rho=1-10^(x[2]-1.6),eps=10^(x[3]-7)); m(dtag))
addopt2(x;ep=1) = (Knet.seed!(1); m=Tagger2(); adadelta!(m,repeat(dtag,ep),lr=10^(x[1]),rho=1-10^(x[2]-1.6),eps=10^(x[3]-7)); m(dtag))

rmsoptz(x;ep=1) = (Knet.seed!(1); m=lenet();   rmsprop!(m,repeat(dtrn,ep),lr=10^(x[1]-2.0),rho=1-10^(x[2]-1.6),eps=10^(x[3]-6)); m(dtrn))
rmsopt0(x;ep=1) = (Knet.seed!(1); m=Tagger0(); rmsprop!(m,repeat(dtag,ep),lr=10^(x[1]-2.0),rho=1-10^(x[2]-1.6),eps=10^(x[3]-6)); m(dtag))
rmsopt1(x;ep=1) = (Knet.seed!(1); m=Tagger1(); rmsprop!(m,repeat(dtag,ep),lr=10^(x[1]-2.0),rho=1-10^(x[2]-1.6),eps=10^(x[3]-6)); m(dtag))
rmsopt2(x;ep=1) = (Knet.seed!(1); m=Tagger2(); rmsprop!(m,repeat(dtag,ep),lr=10^(x[1]-2.0),rho=1-10^(x[2]-1.6),eps=10^(x[3]-6)); m(dtag))

admoptz(x;ep=1) = (Knet.seed!(1); m=lenet();   adam!(m,repeat(dtrn,ep),lr=10^(x[1]-2.5),beta1=1-10^(x[2]-1),beta2=1-10^(x[3]-3),eps=10^(x[4]-6)); m(dtrn))
admopt0(x;ep=1) = (Knet.seed!(1); m=Tagger0(); adam!(m,repeat(dtag,ep),lr=10^(x[1]-2.5),beta1=1-10^(x[2]-1),beta2=1-10^(x[3]-3),eps=10^(x[4]-6)); m(dtag))
admopt1(x;ep=1) = (Knet.seed!(1); m=Tagger1(); adam!(m,repeat(dtag,ep),lr=10^(x[1]-2.5),beta1=1-10^(x[2]-1),beta2=1-10^(x[3]-3),eps=10^(x[4]-6)); m(dtag))
admopt2(x;ep=1) = (Knet.seed!(1); m=Tagger2(); adam!(m,repeat(dtag,ep),lr=10^(x[1]-2.5),beta1=1-10^(x[2]-1),beta2=1-10^(x[3]-3),eps=10^(x[4]-6)); m(dtag))

function doit()
    for (f,n) in (
(sgdoptz, 1),
(sgdopt0, 1),
(sgdopt1, 1),
(sgdopt2, 1),
       
(momoptz, 2),
(momopt0, 2),
(momopt1, 2),
(momopt2, 2),
       
(nesoptz, 2),
(nesopt0, 2),
(nesopt1, 2),
(nesopt2, 2),
       
(adgoptz, 2),
(adgopt0, 2),
(adgopt1, 2),
(adgopt2, 2),
       
(addoptz, 3),
(addopt0, 3),
(addopt1, 3),
(addopt2, 3),
       
(rmsoptz, 3),
(rmsopt0, 3),
(rmsopt1, 3),
(rmsopt2, 3),
       
(admoptz, 4),
(admopt0, 4),
(admopt1, 4),
(admopt2, 4))
        @show (f,n)
        Knet.gc()
        map(println,(goldensection(f, n, dxmin=0.01) for i in 1:3))
    end
end
