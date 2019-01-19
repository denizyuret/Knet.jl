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

sgdoptz(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=lenet();   sgd!(m,repeat(dtrn,ep),lr=10^(x[1]-1.0)); m(dtrn))
sgdopt0(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger0(); sgd!(m,repeat(dtag,ep),lr=10^(x[1]-1.0)); m(dtag))
sgdopt1(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger1(); sgd!(m,repeat(dtag,ep),lr=10^(x[1]-1.0)); m(dtag))
sgdopt2(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger2(); sgd!(m,repeat(dtag,ep),lr=10^(x[1]-1.0)); m(dtag))

momoptz(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=lenet();   momentum!(m,repeat(dtrn,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtrn))
momopt0(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger0(); momentum!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))
momopt1(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger1(); momentum!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))
momopt2(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger2(); momentum!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))

nesoptz(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=lenet();   nesterov!(m,repeat(dtrn,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtrn))
nesopt0(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger0(); nesterov!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))
nesopt1(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger1(); nesterov!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))
nesopt2(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger2(); nesterov!(m,repeat(dtag,ep),lr=10^(x[1]-1.6),gamma=1-10^(x[2]-0.7)); m(dtag))

adgoptz(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=lenet();   adagrad!(m,repeat(dtrn,ep),lr=10^(x[1]-1.75),eps=10^(x[2]-6)); m(dtrn))
adgopt0(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger0(); adagrad!(m,repeat(dtag,ep),lr=10^(x[1]-1.75),eps=10^(x[2]-6)); m(dtag))
adgopt1(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger1(); adagrad!(m,repeat(dtag,ep),lr=10^(x[1]-1.75),eps=10^(x[2]-6)); m(dtag))
adgopt2(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger2(); adagrad!(m,repeat(dtag,ep),lr=10^(x[1]-1.75),eps=10^(x[2]-6)); m(dtag))

addoptz(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=lenet();   adadelta!(m,repeat(dtrn,ep),lr=10^(x[1]),rho=1-10^(x[2]-1.6),eps=10^(x[3]-7)); m(dtrn))
addopt0(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger0(); adadelta!(m,repeat(dtag,ep),lr=10^(x[1]),rho=1-10^(x[2]-1.6),eps=10^(x[3]-7)); m(dtag))
addopt1(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger1(); adadelta!(m,repeat(dtag,ep),lr=10^(x[1]),rho=1-10^(x[2]-1.6),eps=10^(x[3]-7)); m(dtag))
addopt2(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger2(); adadelta!(m,repeat(dtag,ep),lr=10^(x[1]),rho=1-10^(x[2]-1.6),eps=10^(x[3]-7)); m(dtag))

rmsoptz(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=lenet();   rmsprop!(m,repeat(dtrn,ep),lr=10^(x[1]-2.0),rho=1-10^(x[2]-1.6),eps=10^(x[3]-6)); m(dtrn))
rmsopt0(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger0(); rmsprop!(m,repeat(dtag,ep),lr=10^(x[1]-2.0),rho=1-10^(x[2]-1.6),eps=10^(x[3]-6)); m(dtag))
rmsopt1(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger1(); rmsprop!(m,repeat(dtag,ep),lr=10^(x[1]-2.0),rho=1-10^(x[2]-1.6),eps=10^(x[3]-6)); m(dtag))
rmsopt2(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger2(); rmsprop!(m,repeat(dtag,ep),lr=10^(x[1]-2.0),rho=1-10^(x[2]-1.6),eps=10^(x[3]-6)); m(dtag))

admoptz(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=lenet();   adam!(m,repeat(dtrn,ep),lr=10^(x[1]-2.5),beta1=1-10^(x[2]-1),beta2=1-10^(x[3]-3),eps=10^(x[4]-6)); m(dtrn))
admopt0(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger0(); adam!(m,repeat(dtag,ep),lr=10^(x[1]-2.5),beta1=1-10^(x[2]-1),beta2=1-10^(x[3]-3),eps=10^(x[4]-6)); m(dtag))
admopt1(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger1(); adam!(m,repeat(dtag,ep),lr=10^(x[1]-2.5),beta1=1-10^(x[2]-1),beta2=1-10^(x[3]-3),eps=10^(x[4]-6)); m(dtag))
admopt2(x;ep=1) = (Knet.seed!(1); Knet.gc(); m=Tagger2(); adam!(m,repeat(dtag,ep),lr=10^(x[1]-2.5),beta1=1-10^(x[2]-1),beta2=1-10^(x[3]-3),eps=10^(x[4]-6)); m(dtag))

function doit()
    for (f,n) in (
#(sgdoptz, 1),
#(sgdopt0, 1),
#(sgdopt1, 1),
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
        map(println,(goldensection(f, n, dxmin=0.01) for i in 1:3))
    end
end

# (f, n) = (sgdoptz, 1)
# (0.06408782f0, [0.381966])
# (0.057187945f0, [0.673762])
# (0.063954495f0, [0.416408])
# (f, n) = (sgdopt0, 1)
# (0.42343873f0, [1.67376])
# (0.42270383f0, [1.67376])
# (0.42387152f0, [1.67376])
# (f, n) = (sgdopt1, 1)
# (2.6929355f0, [0.201626])
# (2.6909966f0, [0.201626])
# (2.6229603f0, [0.124612])
# (f, n) = (sgdopt2, 1)
# (2.4484448f0, [0.18034])
# (2.4713175f0, [0.18034])
# (2.447972f0, [0.18034])
# (f, n) = (momoptz, 2)
# (0.043797858f0, [0.0901699, -0.63932])
# (0.04545113f0, [0.0557281, -0.567412])
# (0.045358934f0, [0.0901699, -0.583592])
# (f, n) = (momopt0, 2)
# (0.36315882f0, [1.0, -1.0])
# (0.36127174f0, [1.0, -1.0])
# (0.36120024f0, [1.0, -1.0])
# (f, n) = (momopt1, 2)
# (0.7988566f0, [0.0344419, -1.48832])
# (0.8195181f0, [0.0, -1.56231])
# (0.509364f0, [0.562306, -1.25735])
# (f, n) = (momopt2, 2)
# (0.26094392f0, [1.09017, -1.01316])
# (0.25581595f0, [1.07701, -0.98382])
# (0.2643916f0, [1.07701, -0.978714])
# (f, n) = (nesoptz, 2)
# (0.044829555f0, [-0.0131556, -0.854102])
# (0.046991f0, [1.09017, 0.381966])
# (0.045459703f0, [0.618034, -0.236068])
# (f, n) = (nesopt0, 2)
# (0.30435556f0, [2.0, -0.0344419])
# (0.30613017f0, [2.01316, 0.0])
# (0.30463794f0, [2.0, -0.0344419])
# (f, n) = (nesopt1, 2)
# (0.3268844f0, [1.0, -1.0])
# (0.42414135f0, [0.527864, -1.47214])
# (0.4111404f0, [0.527864, -1.40325])
# (f, n) = (nesopt2, 2)
# (0.25233853f0, [1.0, -1.20163])
# (0.24862657f0, [1.0, -1.23607])
# (0.25119102f0, [1.0, -1.20163])
# (f, n) = (adgoptz, 2)
# (0.04502045f0, [0.0557281, -0.381966])
# (0.044421237f0, [0.0395478, -0.381966])
# (0.044924963f0, [0.0557281, -0.472136])
# (f, n) = (adgopt0, 2)
# (0.2225517f0, [0.708204, -2.94427])
# (0.2228803f0, [0.673762, -3.65248])
# (0.22272015f0, [0.742646, -4.23607])
# (f, n) = (adgopt1, 2)
# (0.2982552f0, [0.381966, 0.36068])
# (0.30067125f0, [0.381966, 0.36068])
# (0.29412064f0, [0.381966, 0.36068])
# (f, n) = (adgopt2, 2)
# (0.37143835f0, [0.0, 0.124612])
# (0.1904666f0, [0.381966, 0.167184])
# (0.19051659f0, [0.381966, 0.167184])
# (f, n) = (addoptz, 3)
# (0.044699326f0, [0.0212862, -0.618034, 0.854102])
# (0.045474563f0, [0.0557281, -0.618034, 0.854102])
# (0.047650065f0, [-0.029336, -0.618034, 1.03444])
# (f, n) = (addopt0, 3)
# (0.25282782f0, [1.8541, 0.257354, -0.763932])
# (0.2594162f0, [1.8541, 0.618034, 1.11022e-16])
# (0.26069543f0, [1.52786, -0.618034, -0.840946])
# (f, n) = (addopt1, 3)
# (0.26783937f0, [1.23607, 0.944272, 0.416408])
# (0.24327363f0, [1.47214, 0.944272, 0.472136])
# (0.25800592f0, [1.23607, 0.798374, 0.0557281])
# (f, n) = (addopt2, 3)
# (0.19610547f0, [1.41641, 1.23607, 1.22291])
# (0.1768873f0, [1.61803, 1.0, 0.381966])
# (0.1891572f0, [1.38197, 0.854102, 0.777088])
# (f, n) = (rmsoptz, 3)
# (0.04614875f0, [-0.236068, 1.0, 0.0])
# (0.04336815f0, [-0.326238, 1.0, 1.27051])
# (0.04034303f0, [-0.472136, 0.798374, 0.0])
# (f, n) = (rmsopt0, 3)
# (0.24454375f0, [0.326238, 0.0, -1.7082])
# (0.24432705f0, [0.381966, 0.381966, -1.61803])
# (0.24480735f0, [0.381966, 0.0, -1.61803])
# (f, n) = (rmsopt1, 3)
# (0.2691703f0, [-0.0182615, 0.145898, -0.618034])
# (0.3108152f0, [-0.0344419, 0.0557281, -0.583592])
# (0.21720295f0, [0.236068, 1.01316, -0.618034])
# (f, n) = (rmsopt2, 3)
# (0.15476494f0, [0.257354, 1.09017, -1.61803])
# (0.14637487f0, [-0.0344419, 0.875388, -2.48832])
# (0.15130551f0, [0.0557281, 0.944272, -1.8541])
# (f, n) = (admoptz, 4)
# (0.03964424f0, [0.0, 0.0, 0.0, 0.145898])
# (0.041202158f0, [-0.0770143, 1.0, -0.618034, 0.0])
# (0.033100996f0, [0.0, -0.236068, -1.20163, 0.854102])
# (f, n) = (admopt0, 4)
# (0.23469947f0, [0.506578, 0.90983, -0.0901699, -0.708204])
# (0.2352164f0, [0.527864, 0.944272, 0.0, -0.381966])
# (0.23521656f0, [0.472136, 0.965558, 0.0, -1.61803])
# (f, n) = (admopt1, 4)
# (0.16712849f0, [0.381966, 0.0, 0.0, 0.236068])
# (0.17034547f0, [0.381966, -0.326238, 0.0, 0.124612])
# (0.16449016f0, [0.381966, 0.0344419, -1.61803, 0.222912])
# (f, n) = (admopt2, 4)
# (0.11277441f0, [0.381966, 0.236068, -0.236068, 0.381966])
# (0.11162108f0, [0.326238, 0.381966, -1.56231, 0.0])
# (0.11101344f0, [0.36068, 0.36068, 0.0, 0.381966])
