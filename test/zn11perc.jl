using HDF5,JLD,KUnet
isdefined(:xtrn) || (@date @load "zn11oparse.jld")
for y in (:ytrn, :ydev, :ytst); @eval $y=full($y); end # This speeds up the accuracy fn
KUnet.gpu(false) # we don't have full gpu impl for perceptron yet

# xtrn1=xtrn[:,1:100000]
# ytrn1=ytrn[:,1:100000]

net = Layer[Perceptron(size(ytrn,1))]
for epoch=1:1000
    @show epoch
    @date train(net, xtrn, ytrn)
    gc()
    @date ztrn = predict(net,xtrn)
    @date zdev = predict(net,xdev)
    @date ztst = predict(net,xtst)
    @show (epoch, accuracy(ytrn,ztrn), accuracy(ydev,zdev), accuracy(ytst,ztst))
end

