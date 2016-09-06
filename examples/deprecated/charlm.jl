module CharLM
using Knet, ArgParse, JLD

function main(args=ARGS)
    global net, vocab, text, data
    s = ArgParseSettings()
    s.description="charlm.jl (c) Deniz Yuret, 2016. This is the character-level language model from a popular blog post 'The Unreasonable Effectiveness of Recurrent Neural Networks' by Andrej Karpathy, 2015. (http://karpathy.github.io/2015/05/21/rnn-effectiveness)"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="if provided, use first file for training, second for dev, others for test.")
        ("--loadfile"; help="initialize model from file")
        ("--savefile"; help="save final model to file")
        ("--bestfile"; help="save best model to file")
        ("--generate"; arg_type=Int; default=0; help="if non-zero generate given number of bytes.")
        ("--nlayer"; arg_type=Int; default=1)
        ("--hidden"; arg_type=Int; default=256)
        ("--embedding"; arg_type=Int; default=256)
        ("--epochs"; arg_type=Int; default=1)
        ("--batchsize"; arg_type=Int; default=128)
        ("--seqlength"; arg_type=Int; default=100)
        ("--decay"; arg_type=Float64; default=0.9)
        ("--lr"; arg_type=Float64; default=1.0)
        ("--gclip"; arg_type=Float64; default=5.0)
        ("--dropout"; arg_type=Float64; default=0.0)
        ("--seed"; arg_type=Int; default=42)
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o)
    o[:seed] > 0 && setseed(o[:seed])
    text = !isempty(o[:datafiles]) ? map(f->readall(f), o[:datafiles]) : cell(0)
    !isempty(text) && info("Chars read: $((map(length,text)...))")
    vocab = o[:loadfile]!=nothing ? load(o[:loadfile], "vocab") : Dict{Char,Int}()
    if !isempty(text)
        if isempty(vocab)
            for t in text, c in t; get!(vocab, c, 1+length(vocab)); end
        else
            for t in text, c in t; haskey(vocab, c) || error("Unknown char $c"); end
        end
    end
    o[:vocabsize] = length(vocab); info("$(o[:vocabsize]) unique chars")
    net = o[:loadfile]!=nothing ? load(o[:loadfile], "net") : compile(:charlm; o...)
    !isempty(text) && bigtrain(net, text, vocab, o)
    o[:generate] > 0 && generate(net, vocab, o)
end

function generate(f, vocab, o)
    alpha = Array(Char, length(vocab))
    for (c,i) in vocab; alpha[i]=c; end
    x=zeros(Float32, length(alpha), 1)
    y=zeros(Float32, length(alpha), 1)
    xi = 1
    for i=1:o[:generate]
        copy!(y, forw(f,x))
        x[xi] = 0
        r = rand(Float32)
        p = 0
        for c=1:length(y)
            p += y[c]
            r <= p && (xi=c; break)
        end
        x[xi] = 1
        print(alpha[xi])
    end
    println()
end

function bigtrain(net, text, vocab, o)
    data = map(t->minibatch(t, vocab; o...), text)
    setp(net, lr=o[:lr])
    loss = zeros(length(data))
    lastloss = bestloss = Inf
    for epoch=1:o[:epochs]
        for d=1:length(data)
            loss[d] = (d==1 ? train(net, data[d], softloss; o...) : test(net, data[d], softloss))
        end
        println((epoch, o[:lr], loss...)); flush(STDOUT)
        if length(data) > 1
            loss[2] > lastloss && (o[:lr] *= o[:decay]; setp(net, lr=o[:lr]))
            lastloss = loss[2]
            if lastloss < bestloss
                bestloss = lastloss
                o[:bestfile]!=nothing && save(o[:bestfile], "net", clean(net), "vocab", vocab)
            end
        end
    end
    o[:savefile]!=nothing && save(o[:savefile], "net", clean(net), "vocab", vocab)
    return bestloss
end

@knet function charlm(x; embedding=0, nlayer=0, vocabsize=0, o...)
    y = wdot(x; out=embedding)
    z = repeat(y; o..., frepeat=:charlayer, nrepeat=nlayer)
    return wbf(z; out=vocabsize, f=:soft)
end

@knet function charlayer(x; dropout=0, hidden=0, o...)
    y = lstm(x; out=hidden)
    return drop(y; pdrop=dropout)
end

function test(f, data, loss)
    sumloss = 0.0
    reset!(f)
    T = length(data)-1
    for t=1:T
        x = data[t]
        ygold = data[t+1]
        ypred = forw(f,x)
        sumloss += loss(ypred, ygold)
    end
    sumloss/T
end

function train(f, data, loss; gclip=0, seqlength=100, dropout=0, o...)
    ystack = cell(0)
    sumloss = 0.0
    reset!(f, keepstate=false)
    T = length(data)-1
    for t=1:T
        x = data[t]
        ygold = data[t+1]
        ypred = sforw(f,x; dropout=(dropout>0))
        sumloss += loss(ypred, ygold)
        push!(ystack,ygold)
        if (t%seqlength == 0 || t==T)
            while !isempty(ystack)
                ygold = pop!(ystack)
                sback(f,ygold,loss)
            end
            g = (gclip > 0 ? gnorm(f) : 0)
            update!(f; gscale=(g > gclip > 0 ? gclip/g : 1))
            reset!(f, keepstate=true)
            # @show (t, sumloss/t)
        end
    end
    sumloss/T
end

# Create minibatches.  Each minibatch is a (vocabsize, batchsize)
# matrix with one-hot columns.  The vocab dictionary maps characters
# to integer indices.  The data array returned will have
# T=|chars|/batchsize minibatches.  The columns of minibatch t refer
# to characters t, t+T, t+2T etc.  During training if x=data[t], then
# y=data[t+1].

function minibatch(chars, vocab; vocabsize=0, batchsize=0, o...)
    data = cell(0)
    T = div(length(chars), batchsize)
    for t=1:T
        d=zeros(Float32, vocabsize, batchsize)
        for b=1:batchsize
            c = vocab[chars[t + (b-1) * T]]
            d[c,b] = 1
        end
        push!(data, d)
    end
    return data
end

# temp workaround: prevents error in running finalizer: ErrorException("auto_unbox: unable to determine argument type")
@gpu atexit(()->(for r in net.reg; r.out0!=nothing && Main.CUDArt.free(r.out0); end))

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)

end # module

# Complete works of Shakespeare:
# http://www.gutenberg.org/files/100/100.txt

# Karpathy blog:
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
#  we will encode each character into a vector using 1-of-k encoding.
#  The RNN is trained with mini-batch Stochastic Gradient Descent. I like to use RMSProp (a per-parameter adaptive learning rate) to stablilize the updates.
#  At test time, we feed a character into the RNN and get a distribution over what characters are likely to come next. We sample from this distribution, and feed it right back in to get the next letter.
#  PaulGraham:  Lets train a 2-layer LSTM with 512 hidden nodes (approx. 3.5 million parameters), and with dropout of 0.5 after each layer. We'll train with batches of 100 examples and truncated backpropagation through time of length 100 characters. 
#  Shakespeare: 3-layer RNN with 512 hidden nodes on each layer
#  temperature? you can also generate an infinite amount of your own samples at different temperatures with the provided code.

# const path = Pkg.dir("Knet/data/100.txt")
# const url = "http://www.gutenberg.org/files/100/100.txt"
# isfile(path) || (info("Downloading $url"); save(get(url), path))

# Parameters to play with:
# lr=0.001
# gclip=0
# network depth=1
# network width=512
# fbias=1 # 0 did not help
# winit=Xavier # Gaussian did not help
# embedding matrix=none
# dropout=none
# keepstate=no
# rmsprop=no

# start score:  	(3.2801345440473475,131202.10162734985,39999)
# keepstate=true: 	(3.2360347558469806,129438.15419912338,39999)
# lr=0.002		(3.217129004007876,128681.94303131104,39999)
# lr=0.005		(3.16123230188065,126446.13084292412,39999)
# lr=0.01		(2.9789575188841657,119155.32179784775,39999)
# lr=0.02		(12.209143830274368,488353.5440671444,39999) blows up need gradient clip
# lr=0.02 gclip=10	(3.253291426058063,130128.40375089645,39999)
# lr=2.0 gclip=1.0	(2.7846112851396305,111381.66679430008,39999)
# lr=2.0 gclip=5.0	(2.842096986512893,113681.0373635292,39999)
# lr=1.0 gclip=5.0	(2.685244952175879,107407.11284208298,39999)
# lr=0.5 gclip=5.0	(2.726828870917801,109070.42800784111,39999)
# lr=1.0 gclip=10.0	(2.8446781882471064,113784.28285169601,39999)
# lr=1.0 gclip=2.0	(2.7840308177804656,111358.44868040085,39999)
# fixing at lr=1.0 gclip=5.0
# H=1024		(2.988540398555683,119538.62740182877,39999)
# H=256			(2.5366065015989308,101461.72345745564,39999) but this is for one epoch not convergence. and train not dev.

# with dev set: lr=1.0 gclip=5.0 (ep,lr,trn,dev)
# H=512	(1,1.0,2.6913729234082586,2.0592177872112076)
# H=256	(1,1.0,2.5404179560464306,1.9690401755821478)
# H=128 (1,1.0,2.452683317178178,1.955850279233761)

# fix H=128 play with embedding E
# E=16 (1,1.0,2.260463515062112,1.8129167987150456)
# E=32 (1,1.0,2.2403026809662343,1.8023061217339227)
# E=64 (1,1.0,2.2276982673749544,1.7939553815922231)
# E=128 (1,1.0,2.21924334593858,1.7907175187843698)

# H=256 E=128 (1,1.0,2.2865155415581584,1.7820057602604338)
# H=E=512 (1,1.0,2.3814222320723157,1.7827530246014816)

# fix H=E=256
# H=E=256 (1,1.0,2.274747170884358,1.7639083232801682)
# 2-layer: (1,1.0,2.814780586749161,2.1773407803894065)
