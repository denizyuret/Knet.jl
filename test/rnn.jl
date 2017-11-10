using Knet
include(Knet.dir("src/rnn.jl"))  # TODO: will be removed after integration

function rnntest(r::RNN, ws, x, hx=nothing, cx=nothing; o...)
    w = cudnnGetRNNParams(r,ws)
    X,B,T = (size(x,i) for i=1:3)
    @assert X == r.inputSize
    ysize = collect(size(x))
    ysize[1] = Int(r.hiddenSize * (r.direction == 1 ? 2 : 1))
    ysize = tuple(ysize...)     # to match ndims(y) to ndims(x)
    H = Int(r.hiddenSize)
    L = Int(r.numLayers * (r.direction == 1 ? 2 : 1))
    if hx==nothing; hx=fill!(similar(x,(H,B,L)),0); end
    @assert size(hx) == (H,B,L)
    if r.mode == 2              # LSTM
        if cx==nothing; cx=fill!(similar(x,(H,B,L)),0); end
        @assert size(cx) == (H,B,L)
    end
    if r.numLayers==1 && r.inputMode==r.direction==r.mode==0
        # ht = ReLU(W_i * x_t + R_i h_t-1 + b_Wi + b_Ri)
        # w = (W_i, R_i, b_Wi, b_Ri)
        @assert length(w) == 4
        @assert size(w[1]) == (H,X)
        @assert size(w[2]) == (H,H)
        @assert size(w[3]) == size(w[4]) == (H,)
        ys = []
        h2 = reshape(hx, (H,B*L))
        for t = 1:T
            # TODO: MethodError: no method matching getindex(::Knet.KnetArray{Float64,3}, ::Colon, ::Colon, ::Int64)
            xt = reshape(x[(t-1)*X*B+1:t*X*B], (X,B)) # x[:,:,t]
            @show map(size,(w[1],xt,w[2],h2,w[3],w[4]))
            h2 = relu.(w[1] * xt .+ w[2] * h2 .+ w[3] .+ w[4])
            push!(ys, h2)
        end
        hy = reshape(h2, (H,B,L))
        cy = nothing
        y = reshape(hcat(ys...), ysize)
    else
        error("RNN not supported")
    end
    return (y,hy,cy,nothing)
end

dt = Float64
X,H,B,T = 2,3,4,5
(r,w) = rnninit(X,H; dataType=dt, mode=0, binit=zeros)
ka = KnetArray
x1 = ka(randn(dt,X))
x2 = ka(randn(dt,X,B))
x3 = ka(randn(dt,X,B,T))

# foo(w,x,r)=sum(rnn(r,w,x)[1])
# @show gradcheck(foo,w,x1,r; verbose=true,rtol=0.1)
# @show gradcheck(foo,w,x2,r; verbose=true,rtol=0.1)
# @show gradcheck(foo,w,x3,r; verbose=true,rtol=0.1)

r1 = rnn(r,w,x1)
r2 = rnntest(r,w,x1)
r3 = rnn(r,w,x2)
