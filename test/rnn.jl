# TODO:
# - test bidirectional rnns
# - test new interface
# - test keepstate

include("header.jl")
using Knet: rnntest

if gpu() >= 0; @testset "rnn" begin

    Knet.seed!(1)
    eq(a,b)=all(map((x,y)->(x==y==nothing || isapprox(x,y)),a,b))
    gchk(a...)=gradcheck(a...; rtol=0.1, atol=0.05, args=1)
    gnew(a...; o...)=gcheck(a...; kw=o, rtol=0.1, atol=0.05)
    rnn1(p,r,b=nothing)=rnnforw(r,p...;batchSizes=b)[1]
    D,X,H,B,T = Float64,32,32,16,8 # Keep X==H to test skipInput

    r=w=x1=x2=x3=hx1=cx1=hx2=cx2=hx3=cx3=nothing
    rcpu=wcpu=x1cpu=x2cpu=x3cpu=hx1cpu=cx1cpu=hx2cpu=cx2cpu=hx3cpu=cx3cpu=nothing

    for M=(:relu,:tanh,:lstm,:gru), L=1:2, I=(:false,:true), BI=(:false,:true)
        # println((:rnninit,X,H,:dataType,D, :rnnType,M, :numLayers,L, :skipInput,I, :bidirectional,BI, :binit,xavier))

        (r,w) = rnninit(X, H; dataType=D, rnnType=M, numLayers=L, skipInput=I, bidirectional=BI, binit=xavier) # binit=zeros does not pass gchk
        (rcpu,wcpu) = rnninit(X, H; dataType=D, rnnType=M, numLayers=L, skipInput=I, bidirectional=BI, binit=xavier, usegpu=false)
        @test eltype(wcpu) == eltype(w)
        @test size(wcpu) == size(w)
        wcpu = Array(w)
        HL = BI ? 2L : L
        BT = B*T

        # rnntest tests cudnn vs my implementation on gpu
        # rnnforw(rcpu...) compares cpu vs gpu
        x1cpu = randn(D,X); x1 = ka(x1cpu)
        hx1cpu = randn(D,H,1,HL); hx1 = ka(hx1cpu)
        cx1cpu = randn(D,H,1,HL); cx1 = ka(cx1cpu)
        @test eq(rnnforw(r,w,x1),rnntest(r,w,x1))
        @test eq(rnnforw(r,w,x1),rnnforw(rcpu,wcpu,x1cpu))
        @test eq(rnnforw(r,w,x1,hx1,cx1),rnntest(r,w,x1,hx1,cx1))
        @test eq(rnnforw(r,w,x1,hx1,cx1),rnnforw(rcpu,wcpu,x1cpu,hx1cpu,cx1cpu))
        @test eq(rnnforw(r,w,x1,hx1,cx1;batchSizes=[1]),rnntest(r,w,x1,hx1,cx1))
        @test gchk(rnn1,[w,x1,hx1,cx1],r)
        @test gchk(rnn1,[w,x1,hx1,cx1],r,[1])
        @test gchk(rnn1,[wcpu,x1cpu,hx1cpu,cx1cpu],rcpu)
        @test gchk(rnn1,[w,x1],r)
        @test gchk(rnn1,[w,x1],r,[1])
        @test gchk(rnn1,[wcpu,x1cpu],rcpu)
        # @test gchk(rnn1,[wcpu,x1cpu,hx1cpu,cx1cpu],rcpu,[1]) # TODO

        x2cpu =  randn(D,X,B); x2 = ka(x2cpu)
        hx2cpu = randn(D,H,B,HL); hx2 = ka(hx2cpu)
        cx2cpu = randn(D,H,B,HL); cx2 = ka(cx2cpu)
        @test eq(rnnforw(r,w,x2),rnntest(r,w,x2))
        @test eq(rnnforw(r,w,x2),rnnforw(rcpu,wcpu,x2cpu))
        @test eq(rnnforw(r,w,x2,hx2,cx2),rnntest(r,w,x2,hx2,cx2))
        @test eq(rnnforw(r,w,x2,hx2,cx2),rnnforw(rcpu,wcpu,x2cpu,hx2cpu,cx2cpu))
        @test eq(rnnforw(r,w,x2,hx2,cx2;batchSizes=[B]),rnntest(r,w,x2,hx2,cx2))
        @test gchk(rnn1,[w,x2,hx2,cx2],r)
        @test gchk(rnn1,[wcpu,x2cpu,hx2cpu,cx2cpu],rcpu)
        @test gchk(rnn1,[w,x2],r)
        @test gchk(rnn1,[wcpu,x2cpu],rcpu)
        for b in ([B],[B÷2,B÷2],[B÷2,B÷4,B÷4])
            hx2 = ka(randn(D,H,b[1],HL))
            cx2 = ka(randn(D,H,b[1],HL))
            @test gchk(rnn1,[w,x2,hx2,cx2],r,b)
            @test gchk(rnn1,[w,x2],r,b)
            #@test gchk(rnn1,[wcpu,x2cpu,hx2cpu,cx2cpu],rcpu,b) # TODO
        end

        x3cpu = randn(D,X,B,T); x3 = ka(x3cpu)
        hx3cpu = randn(D,H,B,HL); hx3 = ka(hx3cpu)
        cx3cpu = randn(D,H,B,HL); cx3 = ka(cx3cpu)
        @test eq(rnnforw(r,w,x3),rnntest(r,w,x3))
        @test eq(rnnforw(r,w,x3),rnnforw(rcpu,wcpu,x3cpu))
        @test eq(rnnforw(r,w,x3,hx3,cx3),rnntest(r,w,x3,hx3,cx3))
        @test eq(rnnforw(r,w,x3,hx3,cx3),rnnforw(rcpu,wcpu,x3cpu,hx3cpu,cx3cpu))
        @test eq(rnnforw(r,w,x3,hx3,cx3;batchSizes=[B for t=1:T]),rnntest(r,w,x3,hx3,cx3))
        @test gchk(rnn1,[w,x3,hx3,cx3],r)
        @test gchk(rnn1,[wcpu,x3cpu,hx3cpu,cx3cpu],rcpu)
        for b in ([BT],[BT÷2,BT÷2],[BT÷2,BT÷4,BT÷4])
            hx3a = ka(randn(D,H,b[1],HL))
            cx3a = ka(randn(D,H,b[1],HL))
            @test gchk(rnn1,[w,x3,hx3a,cx3a],r,b)
            @test gchk(rnn1,[w,x3],r,b)
            # @test gchk(rnn1,[wcpu,x3cpu,hx3cpu,cx3cpu],rcpu,b) # TODO
        end

        # new interface
        rnew = RNN(X, H; dataType=D, rnnType=M, numLayers=L, skipInput=I, bidirectional=BI, binit=xavier)
        rnew.w = param(w)

        yold1 = rnnforw(r,w,x3)
        ynew1 = rnew(x3)
        @test isapprox(yold1[1], ynew1)
        @test gnew(rnew, x3)

        yold2 = rnnforw(r,w,x3,hx3,cx3)
        hnew2 = Any[hx3,cx3]
        ynew2 = rnew(x3, hidden=hnew2)
        @test isapprox(yold2[1], ynew2)
        @test isapprox(yold2[2], hnew2[1])
        if M == :lstm
            @test isapprox(yold2[3], hnew2[2])
        end
        #TODO: @test gnew(rnew, x3; hidden=Any[hx3,cx3])

        bs = [B for t=1:T]
        yold3 = rnnforw(r,w,x3,hx3,cx3; batchSizes=bs)
        hnew3 = Any[hx3,cx3]
        ynew3 = rnew(x3, hidden=hnew3, batchSizes=bs)
        @test isapprox(yold3[1], ynew3)
        @test isapprox(yold3[2], hnew3[1])
        if M == :lstm
            @test isapprox(yold3[3], hnew3[2])
        end
        #TODO: @test gnew(rnew, x3; hidden=Any[hx3,cx3], batchSizes=bs)

        #TODO: compare result and diff between multi-step and single-step

        # rnnparam, rnnparams
        for m in (1,2)
            for l in 1:L
                for i in 1:(M==:lstm ? 8 : M==:gru ? 6 : 2)
                    #@show M,L,I,l,i,m
                    @test rnnparam(r,w,l,i,m) == rnnparam(r,wcpu,l,i,m)
                end
            end
        end
        @test all(map(==, rnnparams(r,w), rnnparams(r,wcpu)))
    end # for
end # @testset begin

end # if gpu() >= 0

nothing
