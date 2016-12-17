for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet
!isdefined(:MNIST) && (local lo=isdefined(:load_only); load_only=true; include(Knet.dir("examples","mnist.jl")); load_only=lo)


"""

This example demonstrates the usage of stochastic gradient descent(sgd) based
optimization methods. We train LeNet model on MNIST dataset similar to `lenet.jl`.

You can run the demo using `julia optimizers.jl`.  Use `julia optimizers.jl
--help` for a list of options. By default the [LeNet](http://yann.lecun.com/exdb/lenet)
convolutional neural network model will be trained using sgd for 10 epochs.
At the end of the training accuracy for the training and test sets for each epoch will be printed 
and optimized parameters will be returned.

"""
module Optimizers
using Knet,ArgParse
using Main.MNIST: minibatch, accuracy, xtrn, ytrn, xtst, ytst

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="optimizers.jl (c) Ozan Arkan Can and Deniz Yuret, 2016. Demonstration of different sgd based optimization methods using LeNet."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=100; help="minibatch size")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
	("--eps"; arg_type=Float64; default=1e-6; help="epsilon parameter used in adam, adagrad, adadelta")
	("--gamma"; arg_type=Float64; default=0.95; help="gamma parameter used in momentum")
	("--rho"; arg_type=Float64; default=0.9; help="rho parameter used in adadelta and rmsprop")
	("--beta1"; arg_type=Float64; default=0.9; help="beta1 parameter used in adam")
	("--beta2"; arg_type=Float64; default=0.95; help="beta2 parameter used in adam")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs for training")
	("--optim"; default="Sgd"; help="optimization method (Sgd, Momentum, Adam, Adagrad, Adadelta, Rmsprop)")
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    gpu() >= 0 || error("LeNet only works on GPU machines.")

    dtrn = minibatch4(xtrn, ytrn, o[:batchsize])
    dtst = minibatch4(xtst, ytst, o[:batchsize])
    w = weights()
    prms = params(w, o)
    
    log = Any[]
    report(epoch)=push!(log, (:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))

    report(0)
    @time for epoch=1:o[:epochs]
	    train(w, prms, dtrn; lr=o[:lr], epochs=1)
	    report(epoch)
    end

    for t in log; println(t); end
    return w
end

function train(w, prms, data; lr=.1, epochs=20, nxy=0)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(w, x, y)
            for i in 1:length(w)
		    w[i], prms[i] = update!(w[i], g[i], prms[i])
            end
        end
    end
    return w
end

function predict(w,x,n=length(w)-4)
    for i=1:2:n
        x = pool(relu(conv4(w[i],x; padding=0) .+ w[i+1]))
    end
    x = mat(x)
    for i=n+1:2:length(w)-2
        x = relu(w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+ w[end]
end

function loss(w,x,ygold)
    ypred = predict(w,x)
    ynorm = logp(ypred,1)  # ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function weights(;ftype=Float32,atype=KnetArray)
    w = Array(Any,8)
    w[1] = xavier(Float32,5,5,1,20)
    w[2] = zeros(Float32,1,1,20,1)
    w[3] = xavier(Float32,5,5,20,50)
    w[4] = zeros(Float32,1,1,50,1)
    w[5] = xavier(Float32,500,800)
    w[6] = zeros(Float32,500,1)
    w[7] = xavier(Float32,10,500)
    w[8] = zeros(Float32,10,1)
    return map(a->convert(atype,a), w)
end

#Creates necessary parameters for each weight to use in the optimization
function params(ws, o)
	prms = Any[]
	
	for i=1:length(ws)
		w = ws[i]
		if o[:optim] == "Sgd"
			prm = Sgd(;lr=o[:lr])
		elseif o[:optim] == "Momentum"
			prm = Momentum(w; lr=o[:lr], gamma=o[:gamma])
		elseif o[:optim] == "Adam"
			prm = Adam(w; lr=o[:lr], beta1=o[:beta1], beta2=o[:beta2], eps=o[:eps])
		elseif o[:optim] == "Adagrad"
			prm = Adagrad(w; lr=o[:lr], eps=o[:eps])
		elseif o[:optim] == "Adadelta"
			prm = Adadelta(w; lr=o[:lr], rho=o[:rho], eps=o[:eps])
		elseif o[:optim] == "Rmsprop"
			prm = Rmsprop(w; lr=o[:lr], rho=o[:rho], eps=o[:eps])
		else
			error("Unknown optimization method!")
		end
		push!(prms, prm)
	end

	return prms
end

function minibatch4(x, y, batchsize; atype=KnetArray{Float32})
    data = minibatch(x,y,batchsize; atype=atype)
    for i=1:length(data)
        (x,y) = data[i]
        data[i] = (reshape(x, (28,28,1,batchsize)), y)
    end
    return data
end

function xavier(a...)
    w = rand(a...)
     # The old implementation was not right for fully connected layers:
     # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
        fanin = div(length(w), fanout)
    end
    # See: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
end


# This allows both non-interactive (shell command) and interactive calls like:
# $ julia optimizers.jl --epochs 10
# julia> Optim.main("--epochs 10")
!isinteractive() && (!isdefined(Main,:load_only) || !Main.load_only) && main(ARGS)

end # module

#=
Example Runs
julia optimizers.jl --epochs 10 --lr 0.1
(:epoch,0,:trn,(0.116283335f0,2.2941937f0),:tst,(0.1145f0,2.2946365f0))
(:epoch,1,:trn,(0.9543667f0,0.13499537f0),:tst,(0.958f0,0.12281874f0))
(:epoch,2,:trn,(0.9761f0,0.07270251f0),:tst,(0.9777f0,0.068315394f0))
(:epoch,3,:trn,(0.9838667f0,0.050217737f0),:tst,(0.9827f0,0.051610798f0))
(:epoch,4,:trn,(0.98865f0,0.035738606f0),:tst,(0.9861f0,0.041292988f0))
(:epoch,5,:trn,(0.99125f0,0.027983023f0),:tst,(0.9878f0,0.03583622f0))
(:epoch,6,:trn,(0.99266666f0,0.022933502f0),:tst,(0.9887f0,0.03300085f0))
(:epoch,7,:trn,(0.994f0,0.019440753f0),:tst,(0.9892f0,0.031878017f0))
(:epoch,8,:trn,(0.99455f0,0.017081417f0),:tst,(0.9896f0,0.032124583f0))
(:epoch,9,:trn,(0.9952f0,0.015017059f0),:tst,(0.9895f0,0.032097496f0))
(:epoch,10,:trn,(0.9956333f0,0.013202912f0),:tst,(0.99f0,0.032265987f0))

julia optimizers.jl --epochs 10 --optim Momentum --lr 0.005 --gamma 0.99
(:epoch,0,:trn,(0.1113f0,2.3082018f0),:tst,(0.109f0,2.3084798f0))
(:epoch,1,:trn,(0.9766167f0,0.0770328f0),:tst,(0.9771f0,0.07414803f0))
(:epoch,2,:trn,(0.9861f0,0.043933548f0),:tst,(0.9827f0,0.053114332f0))
(:epoch,3,:trn,(0.98613334f0,0.04348948f0),:tst,(0.984f0,0.057768427f0))
(:epoch,4,:trn,(0.9903167f0,0.030606346f0),:tst,(0.9874f0,0.049403645f0))
(:epoch,5,:trn,(0.99516666f0,0.015177544f0),:tst,(0.9896f0,0.037613332f0))
(:epoch,6,:trn,(0.9928667f0,0.022532726f0),:tst,(0.9874f0,0.04748415f0))
(:epoch,7,:trn,(0.99306667f0,0.020087594f0),:tst,(0.9865f0,0.046911795f0))
(:epoch,8,:trn,(0.9956333f0,0.013521209f0),:tst,(0.9897f0,0.04430927f0))
(:epoch,9,:trn,(0.99703336f0,0.009070589f0),:tst,(0.9896f0,0.04142135f0))
(:epoch,10,:trn,(0.99545f0,0.013547439f0),:tst,(0.9889f0,0.050601155f0))

julia optimizers.jl --epochs 10 --optim Adam --lr 0.001 --beta1 0.9 --beta2 0.95 --eps 1e-8
(:epoch,0,:trn,(0.09903333f0,2.3000493f0),:tst,(0.1008f0,2.2998705f0))
(:epoch,1,:trn,(0.9783667f0,0.068403654f0),:tst,(0.9778f0,0.06494001f0))
(:epoch,2,:trn,(0.99105f0,0.029772233f0),:tst,(0.9881f0,0.035110194f0))
(:epoch,3,:trn,(0.9928667f0,0.022084517f0),:tst,(0.9888f0,0.03618613f0))
(:epoch,4,:trn,(0.9949833f0,0.016130717f0),:tst,(0.99f0,0.035175573f0))
(:epoch,5,:trn,(0.9946f0,0.016869426f0),:tst,(0.9911f0,0.037593916f0))
(:epoch,6,:trn,(0.99328333f0,0.020982243f0),:tst,(0.9896f0,0.04647621f0))
(:epoch,7,:trn,(0.99435f0,0.018795041f0),:tst,(0.9883f0,0.056620233f0))
(:epoch,8,:trn,(0.99576664f0,0.014459096f0),:tst,(0.9896f0,0.053490806f0))
(:epoch,9,:trn,(0.99726665f0,0.0086101545f0),:tst,(0.9902f0,0.053602222f0))
(:epoch,10,:trn,(0.99805f0,0.0059931586f0),:tst,(0.9907f0,0.04834718f0))

julia optimizers.jl --epochs 10 --optim Adagrad --lr 0.01 --eps 1e-6
(:epoch,0,:trn,(0.09983333f0,2.3135705f0),:tst,(0.102f0,2.3134475f0))
(:epoch,1,:trn,(0.98385f0,0.052541837f0),:tst,(0.9822f0,0.051062915f0))
(:epoch,2,:trn,(0.98948336f0,0.03411366f0),:tst,(0.9874f0,0.03760465f0))
(:epoch,3,:trn,(0.99258333f0,0.024431698f0),:tst,(0.9883f0,0.031564854f0))
(:epoch,4,:trn,(0.9945667f0,0.018760378f0),:tst,(0.9895f0,0.028625559f0))
(:epoch,5,:trn,(0.99558336f0,0.015392832f0),:tst,(0.9905f0,0.027373016f0))
(:epoch,6,:trn,(0.99635f0,0.013078845f0),:tst,(0.9907f0,0.026836984f0))
(:epoch,7,:trn,(0.9967f0,0.011378325f0),:tst,(0.991f0,0.026760537f0))
(:epoch,8,:trn,(0.9972f0,0.009899871f0),:tst,(0.9916f0,0.026756253f0))
(:epoch,9,:trn,(0.9975833f0,0.008612509f0),:tst,(0.9917f0,0.026813095f0))
(:epoch,10,:trn,(0.9981167f0,0.0075078686f0),:tst,(0.9914f0,0.026876792f0))

julia optimizers.jl --epochs 10 --optim Adadelta --lr 0.35 --rho 0.9 --eps 1e-6
(:epoch,0,:trn,(0.10545f0,2.3033164f0),:tst,(0.1044f0,2.3031325f0))
(:epoch,1,:trn,(0.96865f0,0.094790496f0),:tst,(0.9704f0,0.08684527f0))
(:epoch,2,:trn,(0.98265f0,0.052093666f0),:tst,(0.982f0,0.05110204f0))
(:epoch,3,:trn,(0.98755f0,0.03808583f0),:tst,(0.9854f0,0.041785713f0))
(:epoch,4,:trn,(0.99105f0,0.027597718f0),:tst,(0.9884f0,0.034749445f0))
(:epoch,5,:trn,(0.9936f0,0.020410579f0),:tst,(0.9903f0,0.030838303f0))
(:epoch,6,:trn,(0.99505f0,0.015846655f0),:tst,(0.9916f0,0.029965784f0))
(:epoch,7,:trn,(0.9963167f0,0.011981298f0),:tst,(0.9915f0,0.028378064f0))
(:epoch,8,:trn,(0.99655f0,0.010981782f0),:tst,(0.9906f0,0.029865112f0))
(:epoch,9,:trn,(0.9967667f0,0.009754072f0),:tst,(0.991f0,0.030760523f0))
(:epoch,10,:trn,(0.9967167f0,0.009537837f0),:tst,(0.9907f0,0.033669963f0))

julia optimizers.jl --epochs 10 --optim Rmsprop --lr 0.001 --rho 0.9 --eps 1e-6
(:epoch,0,:trn,(0.0939f0,2.3106394f0),:tst,(0.0982f0,2.3105752f0))
(:epoch,1,:trn,(0.97908336f0,0.064061135f0),:tst,(0.9799f0,0.058447704f0))
(:epoch,2,:trn,(0.9867833f0,0.03991856f0),:tst,(0.9853f0,0.041050147f0))
(:epoch,3,:trn,(0.99165f0,0.026376523f0),:tst,(0.9898f0,0.03340754f0))
(:epoch,4,:trn,(0.9941667f0,0.017389463f0),:tst,(0.9906f0,0.027854504f0))
(:epoch,5,:trn,(0.9946167f0,0.015640112f0),:tst,(0.9901f0,0.031023417f0))
(:epoch,6,:trn,(0.9949333f0,0.014334091f0),:tst,(0.9904f0,0.033757735f0))
(:epoch,7,:trn,(0.99625f0,0.011019098f0),:tst,(0.9907f0,0.034442622f0))
(:epoch,8,:trn,(0.99691665f0,0.008667187f0),:tst,(0.9912f0,0.030480979f0))
(:epoch,9,:trn,(0.9974f0,0.00787889f0),:tst,(0.9906f0,0.03693716f0))
(:epoch,10,:trn,(0.9979f0,0.0062678233f0),:tst,(0.9911f0,0.034750625f0))
=#
