using BenchmarkTools, Knet, CUDA, Printf
pt = BenchmarkTools.prettytime

macro bench(ex)
    quote
        b=@benchmark(($ex; synchronize()), seconds=1)
        c=@benchmark((@diff ($ex)[1]), seconds=1)
        @printf("%s = %s (%d) %s (%d)\n",
                $(sprint(Base.show_unquoted,ex)),
                pt(minimum(b.times)), length(b.times),
                pt(minimum(c.times)), length(c.times))
    end
end

x = param(16,16,128,128)
w = param(3,3,128,128)
y = param(pool(x))
z = param(conv4(w,x))
s = param(100,100)
t = param(100)
a = rand(1:100,100)
b = rand((0,1),100)
c = rand((-1,1),100)

@bench x
@bench x[1]
@bench sum(x)

@bench elu.(x)
@bench relu.(x)
@bench selu.(x)
@bench sigm.(x)

@bench dropout(x,0.5,drop=true)

@bench mat(x)
@bench pool(x)
@bench unpool(y)
@bench conv4(w,x)
@bench deconv4(w,z)

@bench logp(s, dims=1)
@bench softmax(s, dims=1)
@bench logsumexp(s, dims=1)

@bench accuracy(s,a)
@bench nll(s,a)
#@bench bce(t, b)
#@bench logistic(t, c)

# TODO: bmm, rnn
