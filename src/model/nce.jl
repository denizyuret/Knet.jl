immutable NCE <: Model; rnn; tst; trn; psample; prob; alias; noise; kqvec;
    function NCE(rnn::Function; o...)
        rnn = Net(rnn; o...)                            # x[X,B]->h[H,B]
        trn = Net(nce_trn; ninputs=2)                   # h[H,B]->p[B,B],p[K,B]
        tst = Net(nce_tst; o...)                        # h[H,B]->p[V,B]
        new(rnn, tst, trn)
    end
end

params(r::NCE)=vcat(params(r.rnn), params(r.trn))       # trn and tst share parameters
reset!(r::NCE)=map(reset!, (r.rnn, r.trn, r.tst))

@knet function nce_trn(h,r; vocab=0, nce_winit=Gaussian(0,.01), nce_binit=Constant(0))
    w  = par(; init=nce_winit, dims=(vocab,0))
    b  = par(; init=nce_binit, dims=(vocab,))
    rw = dot(r,w)
    rb = dot(r,b)
    y1 = dot(rw,h)
    y2 = add(rb,y1)
end

@knet function nce_tst(h)
    w  = par()
    b  = par()
    y1 = dot(w,h)
    y2 = add(b,y1)
    y3 = soft(y2)
end

function test(m::NCE, data, loss; o...)
    sumloss = numloss = 0
    nce_tst_init(m, data)
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

function train(m::NCE, data, loss=nothing; nsample=0, psample=nothing, gclip=0, gcheck=false, maxnorm=nothing, losscnt=nothing, o...)
    loss==nothing || warn("NCE uses its own loss function for training, the user provided one is ignored.")
    reset!(m)
    nce_alias_init(m, psample, nsample)
    ystack = Any[]
    for item in data
        if item != nothing
            (x,ygold) = item2xy(item)
            isa(ygold, SparseMatrixCSC) || error("NCE expects ygold to be a SparseMatrixCSC")
            h = forw(m.rnn, x...; trn=true, seq=true, o...)
            r = csc2csr(ygold)  # copies the transpose to gpu; TODO: only works on gpu
            y = forw(m.trn, h, r; trn=true, seq=true, o...)
            kqvec = CudaArray(m.kqvec[ygold.rowval]) # TODO: only works on gpu
            push!(ystack, kqvec)
            losscnt != nothing && (losscnt[1] += nce_loss_real(ypred, kqvec); losscnt[2] += 1)

            # TODO: losscnt += 1 is probably not correct here: 15 mins

            nce_alias_gen(m)
            r = csc2csr(m.noise)  # TODO: this only works on gpu
            y = forw(m.trn, h, r; trn=true, seq=true, o...)
            kqvec = CudaArray(m.kqvec[m.noise.rowval]) # TODO: only works on gpu
            push!(ystack, kqvec)
            losscnt != nothing && (losscnt[1] += nce_loss_noise(ypred, m.kqvec[m.noise.rowval]); losscnt[2] += 1)

        else # end of sequence
            while !isempty(ystack)
                # TODO: back needs to give us hgrad but not rgrad: 30 mins
                # TODO: make sure going back our updates are sparse and efficient. 1hr

                noise = pop!(ystack)
                hgrad1 = back(m.trn, noise, nce_loss_noise; seq=true, getdx=1, o...)
                hgrad = copy(hgrad1)
                ygold = pop!(ystack)
                hgrad2 = back(m.trn, ygold, nce_loss_real;  seq=true, getdx=1, o...)
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
        end
    end
end

# m.tst: input, par(w), par(b), dot, add, soft
# m.trn: input(h), input(r), par(w), par(b), dot(rw), dot(rb), dot(rwh), add
function nce_tst_init(m::NCE, data)
    m.tst.out0[1] == nothing || return
    m.trn.out0[1] == nothing && error("must train before test")
    for i=2:3, n=(:op,:out,:out0,:dif,:dif0,:tmp)
        m.tst.(n)[i] = m.trn.(n)[i+1]
    end
end

# See Vose's alias method for sampling from
# http://keithschwarz.com/darts-dice-coins
function nce_alias_init(m::NCE, psample, nsample)
    vocab = length(psample)
    isdefined(m, :noise) && size(m.noise) == (vocab, nsample) || 
    m.noise = spones(length(psample), nsample)
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

function nce_alias_gen(m::NCE)
    (vocab, nsample) = size(m.noise)
    for n=1:nsample
        i = ceil(Int, rand() * vocab)
        w = (rand() <= m.prob[i] ? i : m.alias[i])
        m.noise.rowval[n] = w
    end
end

# TODO: Should we move the loss code into loss.jl?

# J = -s(y)+log(exp s(y)+kq(y))
# ypred is Matrix(B,B), ypred[i,j] is the score of the i'th gold word on the j'th sentence (only ypred[i,i] is used)
# kqvec is Vector(B), kqvec[i] is K * prob(i'th gold word under the noise dist)
function nce_loss_real(ypred::Matrix, kqvec::Vector)
    l = 0.0
    for i=1:size(ypred,1)
        s = ypred[i,i]
        l += -s + log(exp(s)+kqvec[i])
    end
    return l
end

@gpu function nce_loss_real{T}(ypred::CudaMatrix{T}, kqvec::CudaVector{T})
    # kq = CudaArray(size(m.noise, 2) * m.psample[ygold.rowval])
    ytemp = similar(kqvec)
    T <: Float32 ? ccall((:nce_loss_real_32,libknet),Void,(Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), size(ypred,1), ypred, kqvec, ytemp) :
    T <: Float64 ? ccall((:nce_loss_real_64,libknet),Void,(Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}), size(ypred,1), ypred, kqvec, ytemp) :
    error("$T not supported")
    asum(ytemp)
end

@gpu nce_loss_real{T}(ypred::CudaMatrix{T}, kqvec::Vector{T})=nce_loss_real(ypred, CudaArray(kqvec))

# dJ/ds = p(d=1|y)-1 = (exp s(y))/(exp s(y) + kq(y)) - 1 = -(kq(y))/(exp s(y) + kq(y))
function nce_loss_real(ypred::Matrix, kqvec::Vector, ygrad::Matrix)
    fill!(ygrad,0)
    for i=1:size(ygrad,1)
        s = ypred[i,i]
        kq = kqvec[i]
        ygrad[i,i] = -kq / (exp(s) + kq)
    end
end

@gpu function nce_loss_real{T}(ypred::CudaMatrix{T}, kqvec::CudaVector{T}, ygrad::CudaMatrix{T})
    fill!(ygrad,0)
    # kq = CudaArray(size(m.noise,2) * m.psample[ygold.rowval])
    T <: Float32 ? ccall((:nce_grad_real_32,libknet),Void,(Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), size(ypred,1), ypred, kqvec, ygrad) :
    T <: Float64 ? ccall((:nce_grad_real_64,libknet),Void,(Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}), size(ypred,1), ypred, kqvec, ygrad) :
    error("$T not supported")
end

@gpu nce_loss_real{T}(ypred::CudaMatrix{T}, kqvec::Vector{T}, ygrad::CudaMatrix{T})=nce_loss_real(ypred, CudaArray(kqvec), ygrad)

# J = -log(kq(y))+log(exp s(y)+kq(y))
# ypred is (K,B); ypred[k,b] is the score of k'th noise sample in b'th sentence
# kqvec is (K,); kqvec[k] is K * prob(k'th noise sample)
function nce_loss_noise(ypred::Matrix, kqvec::Vector)
    (K,B) = size(ypred)
    J = 0.0
    for k=1:K
        for b=1:B
            s = ypred[k,b]
            kq = kqvec[k]
            J += log(exp(s) + kq) - log(kq)
        end
    end
    return J
end

@gpu function nce_loss_noise{T}(ypred::CudaMatrix{T}, kqvec::CudaVector{T})
    (K,B) = size(ypred)
    ytemp = similar(ypred)
    T <: Float32 ? ccall((:nce_loss_noise_32,libknet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), K, B, ypred, kqvec, ytemp) :
    T <: Float64 ? ccall((:nce_loss_noise_64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}), K, B, ypred, kqvec, ytemp) :
    error("$T not supported")
    asum(ytemp)
end

@gpu nce_loss_noise{T}(ypred::CudaMatrix{T}, kqvec::Vector{T})=nce_loss_noise(ypred,CudaArray(kqvec))

# dJ/ds = (exp s(y))/(exp s(y) + kq(y))
function nce_loss_noise(ypred::Matrix, kqvec::Vector, ygrad::Matrix)
    (K,B) = size(ypred)
    for k=1:K
        for b=1:B
            exps = exp(ypred[k,b])
            ygrad[k,b] = exps/(exps+kqvec[k])
        end
    end
    return ygrad
end

@gpu function nce_loss_noise{T}(ypred::CudaMatrix{T}, kqvec::CudaVector{T}, ygrad::CudaMatrix{T})
    (K,B) = size(ypred)
    T <: Float32 ? ccall((:nce_grad_noise_32,libknet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}), K, B, ypred, kqvec, ygrad) :
    T <: Float64 ? ccall((:nce_grad_noise_64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}), K, B, ypred, kqvec, ygrad) :
    error("$T not supported")
    return ygrad
end

@gpu nce_loss_noise{T}(ypred::CudaMatrix{T}, kqvec::Vector{T}, ygrad::CudaMatrix{T})=nce_loss_noise(ypred,CudaArray(kqvec),ygrad)

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
