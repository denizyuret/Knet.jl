
using Knet, MLDatasets

# throw an error if p2p is not available
@assert enableP2P() "No p2p access"
@assert (nthreads() == gpuCount()) "There is not a thread for each device"

# Returns a tuple of two tuples: training and test data and labels
function cifar10(dir=nothing, onehot=true; dtype = Float32)
    dir = (dir == nothing) ? string(pwd(),"/cifar10") : dir
    loader = MLDatasets.CIFAR10
    (xtr, ytr) = loader.traindata(dir)
    (xts, yts) = loader.testdata(dir)
    xtr = convert(Array{dtype}, xtr)
    xts = convert(Array{dtype}, xts)
    if onehot
        ytr = toonehot(ytr+1, 10)
        yts = toonehot(yts+1, 10)
    end
    return ((xtr, ytr), (xts, yts))
end

function toonehot(ytrnraw, numclass; dtype=Float32)
    yonehot = zeros(dtype, numclass, length(ytrnraw))
    # println(ytrnraw)
    for (i, y) in enumerate(ytrnraw)
      # println(i," ", y)
      yonehot[y, i] = 1.0
   end
    #y[ytrnraw[:], 1:length(ytrnraw)] = 1.0
    return yonehot
end

function loaddata()
    println("Loading data...")
    dtr, dts = cifar10()
    println("Data is read...")
    (xtrn, ytrn) = dtr
    (xtst, ytst) = dts
    mnt = mean(xtrn, (1, 2, 4))
    xtrn .-= mnt
    xtst .-= mnt
    return (xtrn, ytrn), (xtst, ytst)
end

function next_batch(x, y, bs)
    batch_indices = rand(1:size(x, 4), bs)
    x, y =  x[:, :, :, batch_indices], y[:, batch_indices]
    # Distribute batch to available devices
    distribute(x), distribute(y)
end

function conv_init(eltype, dims...)
    stdev = sqrt(2.0 / *(dims[1], dims[2], dims[4]))
    stdev .* randn(eltype, dims...)
end

function init_weights(eltype=Float32)
    Any[
        conv_init(eltype, 3, 3,  3,    32),   zeros(eltype, 1, 1, 32, 1),
        conv_init(eltype, 3, 3,  32,   64),   zeros(eltype, 1, 1, 64, 1),
        conv_init(eltype, 3, 3,  64,  128),   zeros(eltype, 1, 1, 128, 1),
        xavier(eltype, 500,   8 * 8 * 128),   zeros(eltype, 500, 1),
        xavier(eltype,  10,           500),   zeros(eltype, 10, 1)
    ]
end

function pred(w, x)
    o = x
    for i = 1:2:length(w)-4
        o = relu(conv4(w[i], o; padding=1) .+ w[i+1])
        if i > 2
            o == pool(o)
        end
    end
    o = w[end-3] * mat(o) .+ w[end-2]
    w[end-1] * relu(o) .+ w[end]
end

result_loss(scores, ygold) = -sum(ygold .* logp(scores, 1)) ./ size(ygold, 2)

loss(w,x,ygold) = result_loss(pred(w, x), ygold)

# TODO: make this multi-gpu
function accuracy(w, dtst)
    dtype = KnetArray{Float32}
    println("Computing Accuracy...")
    ncorrect = 0
    ninstance = 0
    nloss = 0
    nloss_count = 0
    X, Y = dtst
    bsize = 200
    for i = 1:bsize:size(Y,2)
        x = convert(dtype, X[:, :, :, i:i+bsize-1])
        ygold = convert(dtype, Y[:, i:i+bsize-1])
        if i % 1000 == 0
            println("Accuracy iter ", i)
        end
        ypred = pred(w, x)
        nloss += result_loss(ypred, ygold)
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold, 2)
        nloss_count += 1
    end
    println(ncorrect, " ", ninstance," ", nloss, " ", nloss_count)
    return (ncorrect / ninstance, nloss / nloss_count)
end


function train(;iters=10000, bsize=128, pperiod = 500)
    gpu(0) # force main thread to use the first gpu
    w = init_weights()
    opt = [Momentum(lr=.01) for _ in w]
    # distribute the model
    ws, gradfns = replicate(w, loss)
    dtrn, dtst = loaddata()
    w = ws[1]
    for i = 1:iters
        if (i - 1) % 50 == 0
            println("iter: ", i)
        end
        if (i-1) % 250 == 0
            println("Accuracy: ", accuracy(w, dtst))
            println()
        end
        xs, ys = next_batch(dtrn..., bsize)
        grads = parallel_apply(gradfns, Any[ws, xs, ys])
        # Update weights and sync params (very naive)
        for j = 1:length(w)
            # Average the computed gradients
            g = mean(Any[gr[j] for gr in grads])
            # update the param j of the main thread
            update!(w[j], g, opt[j])
            # copy updated param to other devices
            for n = 2:length(ws)
                copy!(ws[n][j], w[j])
            end
        end
    end
end

train()
              


#=if (i-1) % 250 == 0 # debug iteration
println()
# weight debug
if gpuCount() == 2
for j = 1:length(w)
println(mean(KnetArray{Float32}(w[j] .== ws[2][j]))) # should print 1
end
end
println("Iter: ", i)
@time xs, ys = next_batch(dtrn..., 256)
@time grads = parallel_apply(gradfns, Any[ws, xs, ys])
println(mean(grads[1][1]), " ", mean(grads[2][1]))
# Update weights and sync params (very naive)
@time for j = 1:length(w)
g = +([grads[k][j] for k = 1:length(grads)]...) ./ length(grads)
update!(w[j], g, opt[j])
for n = 2:length(ws)
copy!(ws[n][j], w[j])
end
end
continue
end=#
