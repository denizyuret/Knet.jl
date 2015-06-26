using HDF5,JLD,KUnet
isdefined(:xtrn) || (@date @load "zn11oparse.jld")
for y in (:ytrn, :ydev, :ytst); @eval $y=full($y); end # This speeds up the accuracy fn
KUnet.gpu(false) # we don't have full gpu impl for perceptron yet

# xtrn1=xtrn[:,1:100000]
# ytrn1=ytrn[:,1:100000]

for seed=1:10
    srand(seed)
    net = Layer[Perceptron(size(ytrn,1))]
    for epoch=1:20
        @show epoch
        @date train(net, xtrn, ytrn; shuffle=true)
        @date ztrn = predict(net,xtrn)
        @date zdev = predict(net,xdev)
        @date ztst = predict(net,xtst)
        @show (seed, epoch, accuracy(ytrn,ztrn), accuracy(ydev,zdev), accuracy(ytst,ztst))
    end
end


