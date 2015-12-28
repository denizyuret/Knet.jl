using Knet,HTTPClient

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
data = readdlm(get(url).body.data) # TODO: saving as file as in mnist
@show size(data)
x = data[:,1:13]'
y = data[:,14]'
setseed(42)
r = randperm(size(x,2))
xtrn=x[:,r[1:400]]
ytrn=y[:,r[1:400]]
xtst=x[:,r[401:end]]
ytst=y[:,r[401:end]]

@knet function housing(x)       # TODO: keyword args
    w = par(; dims=(1,13), init=Gaussian(0,.01))      # TODO: 0 size unspecified
    b = par(; dims=(1,), init=Constant(0))        # TODO: do we need init? yes.
    return w * x .+ b
end

net = compile(:housing)         # TODO: knet functions as new op
setp(net; lr=0.01)

function test(f, x, y, loss)
    ypred = forw(f,x)
    return loss(ypred, y)
end

function train(f, x, y, loss)
    for i=1:size(x,2)
        forw(net, x[:,i])       # TODO: minibatching
        back(net, y[:,i], loss)
        update!(net)
    end
end

for epoch=1:10                 # TODO: setting lr
    train(net, xtrn, ytrn, quadloss)
    loss = test(net, xtst, ytst, quadloss)
    println((epoch, loss))
end

# file="housing.data"
# isfile(file) || get(url; ostream=file)
# data = readdlm(file)

