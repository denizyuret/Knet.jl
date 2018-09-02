#abstract type Model end  # do we really need a Model type?
#just iterate on params(f)
#Base.iterate(f::Model, s=(params(f),1)) = ((p,i)=s; i<=length(p) ? (p[i],(p,i+1)) : nothing)

atype()=(gpu() >= 0 ? KnetArray{Float32} : Array{Float32})
param(d...; init=xavier, atype=atype())=Param(atype(init(d...)))
param0(d...; atype=atype())=param(d...; init=zeros, atype=atype)

# Keyword argument problem:
# optimizer, loss, model can all take keyword args; how do we specify them through train?
# We can give a constructed optimizer and deepcopy it for each param.
# We don't call model directly, only through loss (because it may need model params for regularization).
# So we pass all unrecognized kwargs to loss and let it sort out.
function train!(model, data::Data; loss=nll, optimizer=SGD(), callback=ncount(length(data)), o...)
    for param in params(model)
        param.opt = deepcopy(optimizer)
    end
    while true
        for (x,y) in data
            J = @diff loss(model,x,y; o...)
            callback(model,x,y,value(J)) || return
            update!(model, J)
        end
    end
end

ncount(n)=((x...)->(n > 0 && (n -= 1; true)))

function update!(model,J::Tape)
    for w in params(model)
        g = gradient(J,w)
        update!(value(w),g,w.opt)
    end
end

import ProgressMeter            # don't want to import update!

mutable struct Training
    whentorecord; datasets; losses; errors; updatecount; progress
    Training(w,ds...)=new(w, ds, [Float64[] for d in ds], [Float64[] for d in ds], 0, ProgressMeter.Progress(w[end],1))
end

function (t::Training)(model,x,y,loss)
    if t.updatecount ∈ t.whentorecord
        ProgressMeter.update!(t.progress, t.updatecount)
        for (data,loss,err) in zip(t.datasets, t.losses, t.errors)
            push!(loss, nll(model,data))
            push!(err, zeroone(model,data))
        end
    end
    t.updatecount += 1
    return t.updatecount <= t.whentorecord[end]
end


# params(f) Based on deepcopy_internal:

params(f) = (ps=Param[]; params_internal(f,ps,IdDict()); ps)

params_internal(p::Param, ps::Vector{Param}, d::IdDict) = if !haskey(d,p); d[p]=true; push!(ps,p); end

params_internal(x::Union{Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,Task},
                ps::Vector{Param}, stackdict::IdDict) = return
params_internal(x::Tuple, ps::Vector{Param}, stackdict::IdDict) =
    for p in x; params_internal(p, ps, stackdict); end

params_internal(x::Module, ps::Vector{Param}, stackdict::IdDict) = return

params_internal(x::String, ps::Vector{Param}, stackdict::IdDict) = return

function params_internal(x::Core.SimpleVector, ps::Vector{Param}, stackdict::IdDict)
    if haskey(stackdict, x)
        return
    end
    stackdict[x] = true
    for p in x; params_internal(x, ps, stackdict); end
end

function params_internal(@nospecialize(x), ps::Vector{Param}, stackdict::IdDict)
    T = typeof(x)::DataType
    nf = nfields(x)
    (isbitstype(T) || nf == 0) && return
    if haskey(stackdict, x)
        return
    end
    if T.mutable
        stackdict[x] = true
    end
    for i in 1:nf
        if isdefined(x,i)
            params_internal(getfield(x,i), ps, stackdict)
        end
    end
end

function params_internal(x::Array, ps::Vector{Param}, stackdict::IdDict)
    if haskey(stackdict, x)
        return
    end
    _params_array_t(x, eltype(x), ps, stackdict)
end

function _params_array_t(@nospecialize(x), T, ps::Vector{Param}, stackdict::IdDict)
    stackdict[x] = true
    if isbitstype(T)
        return
    end
    for i = 1:(length(x)::Int)
        if ccall(:jl_array_isassigned, Cint, (Any, Csize_t), x, i-1) != 0
            xi = ccall(:jl_arrayref, Any, (Any, Csize_t), x, i-1)
            if !isbits(xi)
                xi = params_internal(xi, ps, stackdict)
            end
        end
    end
end

function params_internal(x::Union{Dict,IdDict}, ps::Vector{Param}, stackdict::IdDict)
    if haskey(stackdict, x)
        return
    end
    stackdict[x] = true
    for (k, v) in x
        params_internal(k, ps, stackdict)
        params_internal(v, ps, stackdict)
    end
end
