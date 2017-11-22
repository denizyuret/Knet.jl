# TODO: test bidirectional rnns

include("header.jl")

if gpu() >= 0

using Knet: rnntest

eq(a,b)=all(map((x,y)->(x==y==nothing || isapprox(x,y)),a,b))
gchk(a...)=gradcheck(a...; rtol=0.01)
rnn1(p,r,b=nothing)=rnnforw(r,p...;batchSizes=b)[1]

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

        hx1 = ka(randn(D,H,1,L))
        cx1 = ka(randn(D,H,1,L))
        @test eq(rnnforw(r,w,x1),rnntest(r,w,x1))
        @test eq(rnnforw(r,w,x1,hx1,cx1),rnntest(r,w,x1,hx1,cx1))
        @test eq(rnnforw(r,w,x1,hx1,cx1;batchSizes=[1]),rnntest(r,w,x1,hx1,cx1))
        @test gchk(rnn1,[w,x1,hx1,cx1],r)
        @test gchk(rnn1,[w,x1,hx1,cx1],r,[1])

        hx2 = ka(randn(D,H,B,L))
        cx2 = ka(randn(D,H,B,L))
        @test eq(rnnforw(r,w,x2),rnntest(r,w,x2))
        @test eq(rnnforw(r,w,x2,hx2,cx2),rnntest(r,w,x2,hx2,cx2))
        @test eq(rnnforw(r,w,x2,hx2,cx2;batchSizes=[B]),rnntest(r,w,x2,hx2,cx2))
        @test gchk(rnn1,[w,x2,hx2,cx2],r)
        for b in ([16],[8,8],[10,4,2])
            hx2 = ka(randn(D,H,b[1],L))
            cx2 = ka(randn(D,H,b[1],L))
            @test gchk(rnn1,[w,x2,hx2,cx2],r,b)
        end

        hx3 = ka(randn(D,H,B,L))
        cx3 = ka(randn(D,H,B,L))
        @test eq(rnnforw(r,w,x3),rnntest(r,w,x3))
        @test eq(rnnforw(r,w,x3,hx3,cx3),rnntest(r,w,x3,hx3,cx3))
        @test eq(rnnforw(r,w,x3,hx3,cx3;batchSizes=[B for t=1:T]),rnntest(r,w,x3,hx3,cx3))
        @test gchk(rnn1,[w,x3,hx3,cx3],r)
        for b in ([160],[80,80],[100,40,20])
            hx3 = ka(randn(D,H,b[1],L))
            cx3 = ka(randn(D,H,b[1],L))
            @test gchk(rnn1,[w,x3,hx3,cx3],r,b)
        end
    end
end

end # if gpu() >= 0

nothing

