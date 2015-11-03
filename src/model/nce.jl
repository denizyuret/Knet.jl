type NCE <: Model; rnn; tst; trn1; trn2; rng; psample; prob; alias; noise; kqvec;
    function NCE(rnn::Function; rng=Base.GLOBAL_RNG, o...)
        rnn = Net(rnn; o...)                            # x[X,B]->h[H,B]
        trn1 = Net(nce_trn; o..., ninputs=2)            # h[H,B]->p[B,B]
        trn2 = Net(nce_trn; o..., ninputs=2)            # h[H,B]->p[K,B]
        tst  = Net(nce_tst; o...)                       # h[H,B]->p[V,B]
        trn1.tosave[end] = trn2.tosave[end] = true      # TODO: this should automatically happen if we had a nce layer with back_reads_y=true
        new(rnn, tst, trn1, trn2, rng)
    end
end

params(r::NCE)=vcat(params(r.rnn), params(r.tst))       # trn1,2 and tst share parameters
reset!(r::NCE)=map(reset!, (r.rnn, r.tst, r.trn1, r.trn2))

@knet function nce_trn(h,r; vocab=0, nce_winit=Gaussian(0,.01), nce_binit=Constant(log(1/vocab)), o...)
    w  = par(; init=nce_winit, dims=(vocab,0))
    b  = par(; init=nce_binit, dims=(vocab,1))
    rw = dot(r,w)
    rb = dot(r,b)
    y1 = dot(rw,h)
    y2 = add(rb,y1)
end

@knet function nce_tst(h; o...)
    w  = par()
    b  = par()
    y1 = dot(w,h)
    y2 = add(b,y1)
    y3 = soft(y2)
end

# m.tst: input, par(w), par(b), dot, add, soft
# m.trn1,2: input(h), input(r), par(w), par(b), dot(rw), dot(rb), dot(rwh), add
function nce_init(m::NCE, data)
    m.tst.out0[1] == nothing || return
    (x,ygold) = item2xy(first(data))
    @assert x!=nothing && ygold!=nothing
    # This should allocate all params:
    h = forw(m.rnn,  x...; trn=true, seq=true)
    r = csc2csr(ygold)
    y = forw(m.trn1, h, r; trn=true, seq=true)
    kqvec = CudaArray(m.kqvec[ygold.rowval])
    g = back(m.trn1, kqvec, nce_loss_real; seq=true, getdx=1)
    back(m.rnn, g; seq=true)
    for i=3:4, n=(:op,:out,:out0,:dif,:dif0,:tmp)
        m.trn2.(n)[i]  = m.trn1.(n)[i]
        m.tst.(n)[i-1] = m.trn1.(n)[i]
    end
    m.trn1.params = filter(x->isa(x,Par), m.trn1.op)
    m.trn2.params = filter(x->isa(x,Par), m.trn2.op)
    m.tst.params = filter(x->isa(x,Par), m.tst.op)
end

function test(m::NCE, data, loss; o...)
    sumloss = numloss = 0
    reset!(m)
    for item in data
        if item != nothing
            (x,ygold) = item2xy(item)
            h = forw(m.rnn,  x...; trn=false, seq=true, o...)
            ypred = forw(m.tst, h; trn=false, seq=true, o...)
            sumloss += loss(ypred, ygold); numloss += 1
        else
            reset!(m)
        end
    end
    return sumloss/numloss
end

function train(m::NCE, data, loss=nothing; nsample=0, psample=nothing, gclip=0, gcheck=false, maxnorm=nothing, losscnt=nothing, o...) #DBG: 6022
    loss==nothing || Base.warn_once("NCE uses its own loss function for training, the user provided one is ignored.")
    nsample==0 && (nsample=100; Base.warn_once("Using nsample=100"))
    psample==nothing && (v=m.trn1.op[3].dims[1];psample=fill(Float32(1/v),v); Base.warn_once("Using uniform noise dist"))
    nce_alias_init(m, psample, nsample)
    nce_init(m, data)
    reset!(m)
    ystack = Any[]
    for item in data
        if item != nothing
            (x,ygold) = item2xy(item)
            isa(ygold, SparseMatrixCSC) || error("NCE expects ygold to be a SparseMatrixCSC")
            h = forw(m.rnn, x...; trn=true, seq=true, o...)
            r = csc2csr(ygold)  # copies the transpose to gpu
            y = forw(m.trn1, h, r; trn=true, seq=true, o...)
            kqvec = CudaArray(m.kqvec[ygold.rowval])
            push!(ystack, kqvec)
            losscnt != nothing && (losscnt[1] += nce_loss_real(y, kqvec); losscnt[2] += 1)

            # TODO: losscnt += 1 is probably not correct here: 15 mins

            nce_alias_sample(m)
            r = csc2csr(m.noise)
            y = forw(m.trn2, h, r; trn=true, seq=true, o...)
            kqvec = CudaArray(m.kqvec[m.noise.rowval])
            push!(ystack, kqvec)
            losscnt != nothing && (losscnt[1] += nce_loss_noise(y, kqvec); losscnt[2] += length(kqvec))

        else # end of sequence
            while !isempty(ystack)
                noise = pop!(ystack)
                hgrad1 = back(m.trn2, noise, nce_loss_noise; seq=true, getdx=1, o...)
                hgrad = copy(hgrad1)
                ygold = pop!(ystack)
                hgrad2 = back(m.trn1, ygold, nce_loss_real;  seq=true, getdx=1, o...)
                axpy!(1, hgrad2, hgrad)
                back(m.rnn, hgrad; seq=true, o...)
            end
            gcheck && break
            g = (gclip > 0 || maxnorm!=nothing ? gnorm(m) : 0)
            update!(m; gclip=(g > gclip > 0 ? gclip/g : 0), o...)
            if maxnorm != nothing
                w=wnorm(m)
                w > maxnorm[1] && (maxnorm[1]=w)
                g > maxnorm[2] && (maxnorm[2]=g)
            end
            reset!(m)
            #@show (losscnt[1]/losscnt[2], losscnt..., maxnorm...)
        end
    end
end

# See Vose's alias method for sampling from
# http://keithschwarz.com/darts-dice-coins
function nce_alias_init(m::NCE, psample, nsample)
    vocab = length(psample)
    isdefined(m, :noise) && size(m.noise) == (vocab, nsample) || 
    (m.noise = spones(eltype(psample), length(psample), nsample))
    isdefined(m, :psample) && isequal(m.psample, psample) && return
    m.psample = copy(psample)
    m.kqvec = nsample * psample
    m.prob = similar(psample)
    m.alias = similar(psample, Int)
    small = Int[]
    large = Int[]
    p = vocab * psample
    for i in 1:vocab; push!((p[i] < 1 ? small : large), i); end
    while !isempty(small) && !isempty(large)
        l = pop!(small)
        g = pop!(large)
        m.prob[l] = p[l]
        m.alias[l] = g
        p[g] = (p[g] + p[l]) - 1
        push!((p[g] < 1 ? small : large), g)
    end
    for i in large; m.prob[i]=1; end
    for i in small; m.prob[i]=1; end
end

function nce_alias_sample(m::NCE)
    (vocab, nsample) = size(m.noise)
    for n=1:nsample
        i = ceil(Int, rand(m.rng) * vocab)
        w = (rand(m.rng) <= m.prob[i] ? i : m.alias[i])
        m.noise.rowval[n] = w
    end
end

# TODO: Should we move the loss code into loss.jl?
# TODO: will have to implement masks like softloss?
# TODO: batchsize normalization?

# J = -s(y)+log(exp s(y)+kq(y))
# ypred is Matrix(B,B), ypred[i,j] is the score of the i'th gold word on the j'th sentence (only ypred[i,i] is used)
# kqvec is Vector(B), kqvec[i] is K * prob(i'th gold word under the noise dist)
function nce_loss_real(ypred::Matrix, kqvec::Vector; o...)
    l = 0.0
    n = size(ypred,1)
    for i=1:n
        s = ypred[i,i]
        l += -s + log(exp(s)+kqvec[i])
    end
    return l/n
end

@gpu function nce_loss_real{T}(ypred::CudaMatrix{T}, kqvec::CudaVector{T}; o...)
    # kq = CudaArray(size(m.noise, 2) * m.psample[ygold.rowval])
    ytemp = similar(kqvec)
    T <: Float32 ? ccall((:nce_loss_real_32,libknet),Void,(Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), size(ypred,1), ypred, kqvec, ytemp) :
    T <: Float64 ? ccall((:nce_loss_real_64,libknet),Void,(Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}), size(ypred,1), ypred, kqvec, ytemp) :
    error("$T not supported")
    vecnorm(ytemp,1)/size(ypred,2)
end

@gpu nce_loss_real{T}(ypred::CudaMatrix{T}, kqvec::Vector{T}; o...)=nce_loss_real(ypred, CudaArray(kqvec); o...)

# dJ/ds = p(d=1|y)-1 = (exp s(y))/(exp s(y) + kq(y)) - 1 = -(kq(y))/(exp s(y) + kq(y))
function nce_loss_real(ypred::Matrix, kqvec::Vector, ygrad::Matrix; o...)
    fill!(ygrad,0)
    n=size(ygrad,1)
    for i=1:n
        s = ypred[i,i]
        kq = kqvec[i]
        ygrad[i,i] = -(kq / (exp(s) + kq))/n
    end
    return ygrad
end

@gpu function nce_loss_real{T}(ypred::CudaMatrix{T}, kqvec::CudaVector{T}, ygrad::CudaMatrix{T}; o...)
    fill!(ygrad,0)
    # kq = CudaArray(size(m.noise,2) * m.psample[ygold.rowval])
    T <: Float32 ? ccall((:nce_grad_real_32,libknet),Void,(Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), size(ypred,1), ypred, kqvec, ygrad) :
    T <: Float64 ? ccall((:nce_grad_real_64,libknet),Void,(Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}), size(ypred,1), ypred, kqvec, ygrad) :
    error("$T not supported")
    return ygrad
end

@gpu nce_loss_real{T}(ypred::CudaMatrix{T}, kqvec::Vector{T}, ygrad::CudaMatrix{T}; o...)=nce_loss_real(ypred, CudaArray(kqvec), ygrad; o...)

# J = -log(kq(y))+log(exp s(y)+kq(y))
# ypred is (K,B); ypred[k,b] is the score of k'th noise sample in b'th sentence
# kqvec is (K,); kqvec[k] is K * prob(k'th noise sample)
function nce_loss_noise(ypred::Matrix, kqvec::Vector; o...)
    (K,B) = size(ypred)
    J = 0.0
    for k=1:K
        kq = kqvec[k]
        logkq = log(kq)
        for b=1:B
            s = ypred[k,b]
            J += log(exp(s) + kq) - logkq
        end
    end
    return J/B
end

@gpu function nce_loss_noise{T}(ypred::CudaMatrix{T}, kqvec::CudaVector{T}; o...)
    (K,B) = size(ypred)
    ytemp = similar(ypred)
    T <: Float32 ? ccall((:nce_loss_noise_32,libknet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), K, B, ypred, kqvec, ytemp) :
    T <: Float64 ? ccall((:nce_loss_noise_64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}), K, B, ypred, kqvec, ytemp) :
    error("$T not supported")
    vecnorm(ytemp,1)/B
end

@gpu nce_loss_noise{T}(ypred::CudaMatrix{T}, kqvec::Vector{T}; o...)=nce_loss_noise(ypred,CudaArray(kqvec); o...)

# dJ/ds = (exp s(y))/(exp s(y) + kq(y))
function nce_loss_noise(ypred::Matrix, kqvec::Vector, ygrad::Matrix; o...)
    (K,B) = size(ypred)
    for k=1:K
        for b=1:B
            exps = exp(ypred[k,b])
            ygrad[k,b] = (exps/(exps+kqvec[k]))/B
        end
    end
    return ygrad
end

@gpu function nce_loss_noise{T}(ypred::CudaMatrix{T}, kqvec::CudaVector{T}, ygrad::CudaMatrix{T}; o...)
    (K,B) = size(ypred)
    T <: Float32 ? ccall((:nce_grad_noise_32,libknet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), K, B, ypred, kqvec, ygrad) :
    T <: Float64 ? ccall((:nce_grad_noise_64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}), K, B, ypred, kqvec, ygrad) :
    error("$T not supported")
    return ygrad
end

@gpu nce_loss_noise{T}(ypred::CudaMatrix{T}, kqvec::Vector{T}, ygrad::CudaMatrix{T}; o...)=nce_loss_noise(ypred,CudaArray(kqvec),ygrad; o...)

@gpu csc2csr{T}(x::SparseMatrixCSC{T})=CudaSparseMatrixCSR{T}(CudaArray(convert(Vector{Cint},x.colptr)), CudaArray(convert(Vector{Cint},x.rowval)), CudaArray(x.nzval), (x.n,x.m), convert(Cint,length(x.nzval)), device())

# Define main, test, noise, gold as four separate networks.
# If we leave w,b outside of networks they will get unnecessarily copied.
# More troubling, they will not get automatically updated.
# A lot of net forw and back code will be duplicated.
# If we construct a full nce_trn network how do we share its parameters with nce_tst?
# I think we'll just have to do that and figure nce_tst later.
# In nce_trn it would be nice to split the noise and the real batch steps.
# so we have three possible phases: test, noise, gold.  noise and gold have their own forw/back/loss.
# they share parameters and structure.
# we want to compute h only once and pass it to both noise and gold.
# so noise and gold should be partial nets with input h only (gold also gets ygold).
# how do we share their weights?  figure that out later.
# run test first so w,b initialized, then operate on the others to point to w,b.

# Call nce_trn with r = random one-hot-rows matrix to get noise sample scores
# Call nce_trn with r = ygold' to get real sample scores
# TODO: figure out w/b sharing, can it be a par argument?
# problem is it cannot be initialized at construction, we only get allocation after first sample
# run the first sample through the test network to get real w and b Par's and then point to them.

# DONE: r = generate r for noise samples: 1hr
# keep nsample and psample as training parameters, they should not be stored in the model
# figure out the genious sampling algorithm and implement again
# try to reuse the same sparse csr array to avoid allocation

# Note that we had to create trn1 and trn2 because the r dimensions are different.  Keep that in mind when designing conditional
# compiler and/or minibatch resize.

# DONE: back needs to give us hgrad but not rgrad: 30 mins
# DONE: make sure going back our updates are sparse and efficient. 1hr

# TODO: a lot of things only work on gpu

# TODO: init bias = -log vocab
# TODO: check losscnt
# TODO: check loss functions
# TODO: cannot infer size if batchsize=1
