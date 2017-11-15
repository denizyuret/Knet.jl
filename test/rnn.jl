# TODO: test bidirectional rnns

include("header.jl")

using Knet: RNN, cudnnGetRNNParams

function rnntest(r::RNN, ws, x, hx=nothing, cx=nothing; o...)
    if r.direction == 1; error("bidirectional not implemented yet"); end
    w = cudnnGetRNNParams(r,ws)
    X,B,T = (size(x,i) for i=1:3) # ndims(x) may be 1,2 or 3
    @assert X == r.inputSize
    Y = Int(r.hiddenSize * (r.direction == 1 ? 2 : 1))
    ysize = ntuple(i->(i==1 ? Y : size(x,i)), ndims(x)) # to match ndims(y) to ndims(x)
    H = Int(r.hiddenSize)
    @assert (r.inputMode == 0 || H == X)
    L = Int(r.numLayers * (r.direction == 1 ? 2 : 1))
    hsize = (H,B,L)
    @assert hx == nothing || size(hx) == hsize
    @assert cx == nothing || size(cx) == hsize
    h = hx==nothing ? fill!(similar(x,hsize),0) : hx

    ys = []
    hs = [ h[:,:,l] for l=1:L ]
    if r.mode <= 1
        @assert r.inputMode == 0 || all(w[1:1+r.direction] .== nothing)
        # ht = f(W_i * x_t + R_i h_t-1 + b_Wi + b_Ri)
        f = r.mode == 0 ? relu : tanh
        for t = 1:T
            for l = 1:L
                wx,wh,bx,bh = w[2l-1],w[2l],w[2L+2l-1],w[2L+2l]
                wxt = (l > 1 ? wx' * hs[l-1] : r.inputMode==0 ? wx' * x[:,:,t] : x[:,:,t])
                hs[l] = f.(wxt .+ wh' * hs[l] .+ bx .+ bh)
            end
            push!(ys, hs[L])
        end
    elseif r.mode == 2           # LSTM
        @assert r.inputMode == 0 || all(w[1:4*(1+r.direction)] .== nothing)
        # it = σ(Wixt + Riht-1 + bWi + bRi) 
        # ft = σ(Wfxt + Rfht-1 + bWf + bRf) 
        # ot = σ(Woxt + Roht-1 + bWo + bRo) 
        # c't = tanh(Wcxt + Rcht-1 + bWc + bRc) 
        # ct = ft◦ct-1 + it◦c't 
        # ht = ot◦tanh(ct)
        c = cx==nothing ? fill!(similar(x,hsize),0) : cx
        cs = [ c[:,:,l] for l=1:L ]
        for t = 1:T
            for l = 1:L
                Wi,Wf,Wc,Wo,Ri,Rf,Rc,Ro = w[1+8*(l-1):8l]
                bWi,bWf,bWc,bWo,bRi,bRf,bRc,bRo = w[8L+1+8*(l-1):8L+8l]
                Wixt = (l > 1 ? Wi' * hs[l-1] : r.inputMode==0 ? Wi' * x[:,:,t] : x[:,:,t])
                Wfxt = (l > 1 ? Wf' * hs[l-1] : r.inputMode==0 ? Wf' * x[:,:,t] : x[:,:,t])
                Wcxt = (l > 1 ? Wc' * hs[l-1] : r.inputMode==0 ? Wc' * x[:,:,t] : x[:,:,t])
                Woxt = (l > 1 ? Wo' * hs[l-1] : r.inputMode==0 ? Wo' * x[:,:,t] : x[:,:,t])
                it = sigm.(Wixt .+ Ri' * hs[l] .+ bWi .+ bRi)
                ft = sigm.(Wfxt .+ Rf' * hs[l] .+ bWf .+ bRf)
                ot = sigm.(Woxt .+ Ro' * hs[l] .+ bWo .+ bRo)
                cn = tanh.(Wcxt .+ Rc' * hs[l] .+ bWc .+ bRc)
                cs[l] = ft .* cs[l] .+ it .* cn
                hs[l] = ot .* tanh.(cs[l])
            end
            push!(ys, hs[L])
        end
    elseif r.mode == 3           # GRU
        @assert r.inputMode == 0 || all(w[1:3*(1+r.direction)] .== nothing)
        # rt = σ(Wrxt + Rrht-1 + bWr + bRr)
        # it = σ(Wixt + Riht-1 + bWi + bRu)
        # h't = tanh(Whxt + rt◦(Rhht-1 + bRh) + bWh)
        # ht = (1 - it)◦h't + it◦ht-1
        for t = 1:T
            for l = 1:L
                Wr,Wi,Wh,Rr,Ri,Rh = w[1+6*(l-1):6l]
                bWr,bWi,bWh,bRr,bRi,bRh = w[6L+1+6*(l-1):6L+6l]
                Wrxt = (l > 1 ? Wr' * hs[l-1] : r.inputMode==0 ? Wr' * x[:,:,t] : x[:,:,t])
                Wixt = (l > 1 ? Wi' * hs[l-1] : r.inputMode==0 ? Wi' * x[:,:,t] : x[:,:,t])
                Whxt = (l > 1 ? Wh' * hs[l-1] : r.inputMode==0 ? Wh' * x[:,:,t] : x[:,:,t])
                rt = sigm.(Wrxt .+ Rr' * hs[l] .+ bWr .+ bRr)
                it = sigm.(Wixt .+ Ri' * hs[l] .+ bWi .+ bRi)
                ht = tanh.(Whxt .+ rt .* (Rh' * hs[l] .+ bRh) .+ bWh)
                hs[l] = (1 .- it) .* ht .+ it .* hs[l]
            end
            push!(ys, hs[L])
        end
    else
        error("RNN not supported")
    end
    hy = hx==nothing ? nothing : reshape(hcat(hs...), hsize)
    cy = r.mode != 2 || cx==nothing ? nothing : reshape(hcat(cs...), hsize)
    y = reshape(hcat(ys...), ysize)
    return (y,hy,cy,nothing)
end

ka = KnetArray
eq(a,b)=all(map((x,y)->(x==y==nothing || isapprox(x,y)),a,b))
gchk(a...)=gradcheck(a...; rtol=0.01)
rnn1(p,r)=rnn(r,p...)[1]

D,X,H,B,T = Float64,32,32,16,10
x1 = ka(randn(D,X))
x2 = ka(randn(D,X,B))
x3 = ka(randn(D,X,B,T))
(r,w) = rnninit(X,H;dataType=D)
hx = ka(randn(D,H,B,1))
cx = ka(randn(D,H,B,1))

@testset "rnn" begin
    for M=(:relu,:tanh,:lstm,:gru), L=1:2, I=(:false,:true)
        (r,w) = rnninit(X, H; dataType=D, rnnType=M, numLayers=L, skipInput=I)
        hx = ka(randn(D,H,B,L))
        cx = ka(randn(D,H,B,L))
        @test eq(rnn(r,w,x1),rnntest(r,w,x1))
        @test eq(rnn(r,w,x1;batchSizes=[1]),rnntest(r,w,x1))
        @test gchk(rnn1,[w,x1],r)
        @test eq(rnn(r,w,x2,hx,cx),rnntest(r,w,x2,hx,cx))
        @test eq(rnn(r,w,x2,hx,cx;batchSizes=[B]),rnntest(r,w,x2,hx,cx))
        @test r.mode==2 ? gchk(rnn1,[w,x2,hx,cx],r) : gchk(rnn1,[w,x2,hx],r)
        @test eq(rnn(r,w,x3,hx,cx),rnntest(r,w,x3,hx,cx))
        @test eq(rnn(r,w,x3,hx,cx;batchSizes=[B for t=1:T]),rnntest(r,w,x3,hx,cx))
        @test r.mode==2 ? gchk(rnn1,[w,x3,hx,cx],r) : gchk(rnn1,[w,x3,hx],r)
    end
end

nothing

