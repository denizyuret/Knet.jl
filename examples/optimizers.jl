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
module Optim
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
	("--optim"; default="sgd!"; help="optimization method (sgd!, momentum!, adam!, adagrad!, adadelta!, rmsprop!)")
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
	    train(w, prms, dtrn; lr=o[:lr], epochs=1, opt=eval(parse(o[:optim])))
	    report(epoch)
    end

    for t in log; println(t); end
    return w
end

function train(w, prms, data; lr=.1, epochs=20, nxy=0, opt=sgd!)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(w, x, y)
            for i in 1:length(w)
		    opt(prms[i], w[i], g[i])
            end
        end
    end
    return w
end

function predict(w,x,n=length(w)-4)
    for i=1:2:n
        x = pool(relu(conv4(w[i],x) .+ w[i+1]))
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
		if o[:optim] == "sgd!"
			prm = SgdParams(o[:lr])
		elseif o[:optim] == "momentum!"
			prm = MomentumParams(o[:lr], o[:gamma], convert(typeof(ws[i]), zeros(size(ws[i]))))
		elseif o[:optim] == "adam!"
			prm = AdamParams(o[:lr], o[:beta1], o[:beta2], 1, o[:eps], convert(typeof(ws[i]), zeros(size(ws[i]))), convert(typeof(ws[i]), zeros(size(ws[i]))))
		elseif o[:optim] == "adagrad!"
			prm = AdagradParams(o[:lr], o[:eps], convert(typeof(ws[i]), zeros(size(ws[i]))))
		elseif o[:optim] == "adadelta!"
			prm = AdadeltaParams(o[:lr], o[:rho], o[:eps], convert(typeof(ws[i]), zeros(size(ws[i]))), convert(typeof(ws[i]), zeros(size(ws[i]))))
		elseif o[:optim] == "rmsprop!"
			prm = RmspropParams(o[:lr], o[:rho], o[:eps], convert(typeof(ws[i]), zeros(size(ws[i]))))
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

julia optimizers.jl --epochs 10 --optim sgd! --lr 0.1
(:epoch,0,:trn,(0.08366667f0,2.30814f0),:tst,(0.0834f0,2.3081908f0))
(:epoch,1,:trn,(0.9668f0,0.10429999f0),:tst,(0.9691f0,0.09337928f0))
(:epoch,2,:trn,(0.97865f0,0.06621845f0),:tst,(0.9799f0,0.063090876f0))
(:epoch,3,:trn,(0.98461664f0,0.048656303f0),:tst,(0.9825f0,0.04972162f0))
(:epoch,4,:trn,(0.98831666f0,0.036512565f0),:tst,(0.986f0,0.041371945f0))
(:epoch,5,:trn,(0.99111664f0,0.027710818f0),:tst,(0.9875f0,0.035865538f0))
(:epoch,6,:trn,(0.9928333f0,0.022544786f0),:tst,(0.9884f0,0.03336629f0))
(:epoch,7,:trn,(0.99425f0,0.018732546f0),:tst,(0.9896f0,0.03175358f0))
(:epoch,8,:trn,(0.9949333f0,0.01620303f0),:tst,(0.9897f0,0.03147226f0))
(:epoch,9,:trn,(0.99565f0,0.013639423f0),:tst,(0.9902f0,0.03023542f0))
(:epoch,10,:trn,(0.99645f0,0.011452887f0),:tst,(0.9908f0,0.029485364f0))

julia optimizers.jl --epochs 10 --optim momentum! --lr 0.005 --gamma 0.99
(:epoch,0,:trn,(0.12743333f0,2.3020618f0),:tst,(0.1227f0,2.3008814f0))
(:epoch,1,:trn,(0.97568333f0,0.079913534f0),:tst,(0.9776f0,0.0733804f0))
(:epoch,2,:trn,(0.9853167f0,0.045719735f0),:tst,(0.9847f0,0.048553515f0))
(:epoch,3,:trn,(0.98938334f0,0.03375993f0),:tst,(0.9856f0,0.048157893f0))
(:epoch,4,:trn,(0.9902833f0,0.03003023f0),:tst,(0.9856f0,0.044891715f0))
(:epoch,5,:trn,(0.99083334f0,0.027737048f0),:tst,(0.985f0,0.05616427f0))
(:epoch,6,:trn,(0.9949333f0,0.014662323f0),:tst,(0.9881f0,0.038927965f0))
(:epoch,7,:trn,(0.9922f0,0.027102735f0),:tst,(0.9855f0,0.05384007f0))
(:epoch,8,:trn,(0.9957167f0,0.013452965f0),:tst,(0.989f0,0.043424647f0))
(:epoch,9,:trn,(0.9981f0,0.006325099f0),:tst,(0.9897f0,0.040737938f0))
(:epoch,10,:trn,(0.99791664f0,0.00570748f0),:tst,(0.9898f0,0.047788296f0))

julia optimizers.jl --epochs 10 --optim adam! --lr 0.001 --beta1 0.9 --beta2 0.95 --eps 1e-8
(:epoch,0,:trn,(0.14405f0,2.3073778f0),:tst,(0.1458f0,2.3067157f0))
(:epoch,1,:trn,(0.9831167f0,0.052491322f0),:tst,(0.9828f0,0.05281107f0))
(:epoch,2,:trn,(0.99013335f0,0.030533511f0),:tst,(0.9873f0,0.037371557f0))
(:epoch,3,:trn,(0.99361664f0,0.020557461f0),:tst,(0.9902f0,0.03308907f0))
(:epoch,4,:trn,(0.99215f0,0.022505302f0),:tst,(0.9882f0,0.042942464f0))
(:epoch,5,:trn,(0.9913167f0,0.026418246f0),:tst,(0.9861f0,0.052241005f0))
(:epoch,6,:trn,(0.99515f0,0.014335429f0),:tst,(0.99f0,0.0428939f0))
(:epoch,7,:trn,(0.9957167f0,0.013034481f0),:tst,(0.9882f0,0.054499023f0))
(:epoch,8,:trn,(0.99736667f0,0.0092404f0),:tst,(0.9914f0,0.046298113f0))
(:epoch,9,:trn,(0.9971833f0,0.00905812f0),:tst,(0.9907f0,0.054691363f0))
(:epoch,10,:trn,(0.99803334f0,0.00642125f0),:tst,(0.9906f0,0.050386086f0))

julia optimizers.jl --epochs 10 --optim adagrad! --lr 0.01 --eps 1e-6
(:epoch,0,:trn,(0.077183336f0,2.3105638f0),:tst,(0.0788f0,2.3104973f0))
(:epoch,1,:trn,(0.9802167f0,0.06402554f0),:tst,(0.9796f0,0.059473045f0))
(:epoch,2,:trn,(0.98735f0,0.040810067f0),:tst,(0.9862f0,0.04049752f0))
(:epoch,3,:trn,(0.99083334f0,0.030240871f0),:tst,(0.9877f0,0.033289436f0))
(:epoch,4,:trn,(0.9932333f0,0.0235379f0),:tst,(0.9896f0,0.029167267f0))
(:epoch,5,:trn,(0.9943f0,0.019219194f0),:tst,(0.9904f0,0.027038217f0))
(:epoch,6,:trn,(0.9953f0,0.016342603f0),:tst,(0.9917f0,0.02620696f0))
(:epoch,7,:trn,(0.9959f0,0.014257936f0),:tst,(0.9917f0,0.0259071f0))
(:epoch,8,:trn,(0.9963833f0,0.012739546f0),:tst,(0.9916f0,0.02615357f0))
(:epoch,9,:trn,(0.99675f0,0.011400618f0),:tst,(0.9908f0,0.026445275f0))
(:epoch,10,:trn,(0.9972f0,0.010114651f0),:tst,(0.9908f0,0.026587203f0))

julia optimizers.jl --epochs 10 --optim adadelta! --lr 0.01 --rho 0.9 --eps 1e-6
(:epoch,0,:trn,(0.08523333f0,2.3051379f0),:tst,(0.0823f0,2.3053992f0))
(:epoch,1,:trn,(0.85865f0,0.52424514f0),:tst,(0.8671f0,0.5004709f0))
(:epoch,2,:trn,(0.9033833f0,0.3283085f0),:tst,(0.9119f0,0.3075404f0))
(:epoch,3,:trn,(0.9213f0,0.26187488f0),:tst,(0.9293f0,0.24362734f0))
(:epoch,4,:trn,(0.9335833f0,0.22018401f0),:tst,(0.9393f0,0.20390196f0))
(:epoch,5,:trn,(0.94315f0,0.1897216f0),:tst,(0.9483f0,0.1749004f0))
(:epoch,6,:trn,(0.95028335f0,0.16669717f0),:tst,(0.9542f0,0.15299088f0))
(:epoch,7,:trn,(0.9559f0,0.14892434f0),:tst,(0.9589f0,0.1363006f0))
(:epoch,8,:trn,(0.9600833f0,0.13474126f0),:tst,(0.9627f0,0.12306133f0))
(:epoch,9,:trn,(0.96353334f0,0.12341626f0),:tst,(0.9675f0,0.1125456f0))
(:epoch,10,:trn,(0.9665167f0,0.11409138f0),:tst,(0.9696f0,0.1039427f0))

julia optimizers.jl --epochs 10 --optim rmsprop! --lr 0.001 --rho 0.9 --eps 1e-6
(:epoch,0,:trn,(0.07675f0,2.3045394f0),:tst,(0.078f0,2.3053956f0))
(:epoch,1,:trn,(0.9766833f0,0.071488865f0),:tst,(0.9767f0,0.0676108f0))
(:epoch,2,:trn,(0.98618335f0,0.042054318f0),:tst,(0.9846f0,0.04375127f0))
(:epoch,3,:trn,(0.99081665f0,0.027419291f0),:tst,(0.9882f0,0.034820337f0))
(:epoch,4,:trn,(0.9938833f0,0.019255767f0),:tst,(0.9895f0,0.031569693f0))
(:epoch,5,:trn,(0.9949833f0,0.015288265f0),:tst,(0.9906f0,0.03174566f0))
(:epoch,6,:trn,(0.9946333f0,0.0154640535f0),:tst,(0.9898f0,0.03660226f0))
(:epoch,7,:trn,(0.99625f0,0.010613186f0),:tst,(0.9903f0,0.036426704f0))
(:epoch,8,:trn,(0.99725f0,0.007961936f0),:tst,(0.991f0,0.038267612f0))
(:epoch,9,:trn,(0.99696666f0,0.008677514f0),:tst,(0.9905f0,0.03928403f0))
(:epoch,10,:trn,(0.99761665f0,0.006873286f0),:tst,(0.9895f0,0.04377764f0))
=#
