"""
`replicate(params, loss::Function, devlist::Array{Int, 1}=Array(0:gpuCount()-1), eltype=Float32)`
returns an array of tupples distributed
to gpus.
Only arrays, tuples and dictionaries are supported for storing params (not array of tupples etc.).
"""
function replicate(params, loss::Function, devlist::Array{Int, 1}=Array(0:gpuCount()-1), eltype=Float32)
    ndevs = length(devlist)
    param_reps = replicate(params, devlist, eltype)
    grad_reps = Vector{Function}(ndevs)

    threads(1:ndevs) do t
        initial_dev = gpu()
        gpu(devlist[t])
        # replicate grad
        grad_reps[t] = grad(loss)
        # recover the initial gpu of each thread
        gpu(initial_dev)
    end
    return param_reps, grad_reps
end

"""
`replicate(params, devlist::Array{Int, 1}=Array(0:gpuCount()-1), eltype=Float32)`
returns an array of params replicated to gpus specified in devlist.
Only arrays, tuples and dictionaries are supported for storing params.
"""
function replicate(params, devlist::Array{Int, 1}=Array(0:gpuCount()-1), eltype=Float32)
    ndevs = length(devlist)
    param_reps = Vector{Any}(ndevs)
    threads(1:ndevs) do t
        initial_dev = gpu()
        gpu(devlist[t])
        # distribute params
        if ~isa(params, Associative)
            param_reps[t] = map(KnetArray{eltype}, params)
        else # Dictionary
            param_reps[t] = map(keys(params)) do k
                KnetArray{eltype}(params[k])
            end
        end
        # recover the initial gpu of each thread
        gpu(initial_dev)
    end
    return param_reps
end


"""
`distribute(data, devlist::Array{Int, 1}=Array(0:gpuCount()-1), eltype=Float32; bdim=0)`
Distributes data by equally dividing the bdim. if bdim is 0, then ndims(data) is used as
the bdim.
"""
function distribute(data, devlist::Array{Int, 1}=Array(0:gpuCount()-1), eltype=Float32; bdim=0)
    ndevs = length(devlist)
    if bdim == 0
        bdim = ndims(data)
        batch_size = size(data)[end]
    else
        batch_size = size(data)[bdim]
    end
    rem_size = batch_size % ndevs
    rem_size !== 0 && error("Inequal distribution is not supported yet")
    b_size = div(batch_size, ndevs)
    datas = Vector{Any}(ndevs)
    threads(1:ndevs) do t
        temp = gpu()
        gpu(devlist[t])
        start = b_size * (t-1) + 1
        finish = b_size * t
        ranges = []
        for i = 1:ndims(data)
            if i == bdim
                push!(ranges, start:finish)
            else
                push!(ranges, :)
            end
        end
        datas[t] = KnetArray{eltype}(data[ranges...])
        gpu(temp)
    end
    return datas
end


"""
`parallel_apply(f::Function, args::Array{Any,1}, devlist::Array{Int, 1}=Array(0:gpuCount()-1))`
 Execute the function f in parallel with the argument list
"""
function parallel_apply(f::Function, args::Array{Any,1}, devlist::Array{Int, 1}=Array(0:gpuCount()-1))
    
    ndevs = length(devlist)
    outputs = Vector{Any}(ndevs)
    threads(1:ndevs) do t
        temp = gpu()
        gpu(devlist[t])
        largs = (arg[t] for arg in args)
        outputs[t] = f(largs...)
        gpu(temp)
    end
    return outputs
end


"""
`parallel_apply(f::Function, args::Array{Any,1}, devlist::Array{Int, 1}=Array(0:gpuCount()-1))`
 Execute the function f in parallel with the argument list

"""
function parallel_apply(f::Array{Function, 1},
                        args::Array{Any,1}, devlist::Array{Int, 1}=Array(0:gpuCount()-1))
    
    ndevs = length(devlist)
    outputs = Vector{Any}(ndevs)
    threads(1:ndevs) do t
        temp = gpu()
        gpu(devlist[t])
        largs = (arg[t] for arg in args)
        outputs[t] = f[t](largs...)
        gpu(temp)
    end
    return outputs
end
