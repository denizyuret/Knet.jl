using Knet
#=
 Make sure:
    - There are two threads and two gpus
    - p2p access is granted between gpus
=#
assert(nthreads() == 2 && gpuCount() == 2 && enableP2P())

function init_weights(input::Int, output::Int, std = .1)
    Any[std .* randn(output, input), zeros(output, 1)]
end

pred(w, x) = w[1] * x .+ w[2]
loss(w, x, y) = sumabs2(pred(w, x) .- y) / size(y,2)

function loaddata(test=0.0)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    file=Knet.dir("data","housing.data")
    if !isfile(file)
        info("Downloading $url to $file")
        download(url, file)
    end
    data = readdlm(file)'
    @show size(data) # (14,506)
    x = data[1:13,:]
    y = data[14:14,:]
    x = (x .- mean(x,2)) ./ std(x,2) # Data normalization
    if test == 0
        xtrn = xtst = x
        ytrn = ytst = y
    else
        r = randperm(size(x,2))          # trn/tst split
        n = round(Int, (1-test) * size(x,2))
        xtrn=x[:,r[1:n]]
        ytrn=y[:,r[1:n]]
        xtst=x[:,r[n+1:end]]
        ytst=y[:,r[n+1:end]]
    end
    (xtrn, ytrn, xtst, ytst)
end

function train()
    xtrn, ytrn, _, _ = loaddata()
    inp, out = size(xtrn, 1), size(ytrn, 1)
    #bperd = div(size(xtrn, 2), 2)

    # Distribute data and initialize model replicas
    w = init_weights(inp, out)
    # use utils from parallel.jl to replicate model and distribute data
    xs = distribute(xtrn)
    ys = distribute(ytrn)
    ws, gradfns = replicate(w, loss) # copy the weights and recorder to each gpu
    println(xs, "\n", ys, "\n", ws) 
    # The multi gpu training loop
    lr = .1
    for epoch = 1:25
        println()
        println("epoch: ", epoch)
        # Parallel gradient computation using two threads(1 thread per gpu)
        # gradfn[i] is applied to ws[i], xs[i], ys[i]
        grads = parallel_apply(gradfns, Any[ws, xs, ys])
        w1 = ws[gpu()]
        w2 = ws[(1+gpu())%3]
        #TODO: This loop should be pipelined(using streams etc.)
        # i.e, copy of param i+1 and update of param i can be done in parallel
        for i = 1:length(w1)
            # exploit gpu-direct to accumulate gradients
            g = (grads[1][i] .+ grads[2][i]) ./ 2
            # update the parameter server (i.e parameters of thread 1)
            axpy!(-lr, g, w1[i])
            # sync models using gpu-direct
            copy!(w2[i], w1[i])
        end
        # Measure the loss by distributing forward pass to gpus
        losses = parallel_apply(loss, Any[ws, xs, ys])
        println("Losses: ",losses)
        println("Total loss: ", mean(losses))
    end
end

train()
