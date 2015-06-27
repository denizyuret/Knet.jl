using HDF5,JLD,KUnet
KUnet.gpu(false) # we don't have full gpu impl for perceptron yet
isdefined(:xtrn) || (@date @load "zn11oparse1.jld")
for y in (:ytrn, :ydev, :ytst); @eval $y=full(float64($y)); end # This speeds up the accuracy fn
for x in (:xtrn, :xdev, :xtst); @eval $x=float64($x); end # This prevents early convergence

for seed=0:10
    srand(seed)
    net = Layer[Perceptron(size(ytrn,1))]
    for epoch=1:50
        @show epoch
        @date train(net, xtrn, ytrn; shuffle=(seed>0))
        @date ztrn = predict(net,xtrn)
        @date zdev = predict(net,xdev)
        @date ztst = predict(net,xtst)
        @show (seed, epoch, accuracy(ytrn,ztrn), accuracy(ydev,zdev), accuracy(ytst,ztst))
    end
end


