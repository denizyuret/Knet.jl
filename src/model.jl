export Model, track, back!, getgrad, setgrad!, params

"""

"""
abstract type Model end

"""

"""
track(x::AutoGrad.Rec) = track(x.value)
track(x, tape::AutoGrad.Tape=AutoGrad.Tape()) = AutoGrad.Rec(x, tape)
track(x, y) = x
track(x, y::AutoGrad.Rec) = AutoGrad.Rec(x, y.tapes[])
track(x::AutoGrad.Rec, y) = x.value
track(x::AutoGrad.Rec, y::AutoGrad.Rec) = x in y.tapes[] ? x : track(x.value, y)

"""

"""
function back!(x::AutoGrad.Rec)
    tape = x.tapes[]
    AutoGrad.complete!(tape)

    # copied from AutoGrad.jl/src/core.jl:backward_pass
    for n in tape[end-1:-1:1]
        n.outgrad == nothing && continue
        r = n.rec
        for i=1:length(n.parents)
            isassigned(n.parents,i) || continue
            p = n.parents[i]
            og = r.func(AutoGrad.Grad{i},n.outgrad,r.value,r.args...;r.kwargs...)
            p.outgrad = AutoGrad.sum_outgrads(p.outgrad, og)
        end
    end

    tape[1].outgrad
end

function back!(x::AutoGrad.Rec, Δ)
    setgrad!(x, Δ)
    back!(x)
end

"""

"""
function getgrad(x::AutoGrad.Rec)
    x.nodes[].outgrad
end

"""

"""
function setgrad!(x::AutoGrad.Rec, Δ)
    x.nodes[].outgrad = Δ
end

"""
Get a list of tracked parameters from a model.
A model's paremeters can be tracked by running with a tracked input.
Complex models should override this method for efficiency.
"""
function params(m)
    set = ObjectIdDict()

    traverse(m::AutoGrad.Rec) = set[m] = nothing
    traverse(m) = try
        foreach(traverse, m)
    catch
        for attr in fieldnames(typeof(m))
            traverse(getfield(m, attr))
        end
    end

    traverse(m)

    collect(AutoGrad.Rec, keys(set))
end


export Affine, Chain, LSTM, Embedding

mutable struct Affine <: Model
    weight
    bias
end

Affine(a::Integer, b::Integer; init=xavier) = Affine(init(a, b), init(1, b))

function (m::Affine)(x)
    m.weight = track(m.weight, x)
    m.bias   = track(m.bias, x)
    x * m.weight .+ m.bias
end

mutable struct Chain <: Model
    layers::Vector
end

Chain(x...) = Chain(collect(x))

(m::Chain)(x) = foldl((x, m) -> m(x), x, m.layers)

mutable struct LSTM <: Model
    weight
    bias
end

function LSTM(a::Integer, b::Integer; init=rand)
    LSTM(init(a+b, 4b), zeros(1, 4b))
end

# patch for ambiguity of Base/abstractarray.jl:1067 and AutoGrad/abstractarray.jl:168
Base.hcat(a::AutoGrad.Rec, b::AutoGrad.Rec, c::AutoGrad.Rec...) = AutoGrad.cat(2, a, b, c...)

function (m::LSTM)(x, hidden, cell)
    m.weight = track(m.weight, x)
    m.bias   = track(m.bias, x)

    gates   = [x hidden] * m.weight .+ m.bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    hidden, cell
end

"""
`back!` through Embedding will lose all gradients
"""
mutable struct Embedding <: Model
    mat
end

function Embedding(a::Integer, b::Integer; init=rand)
    Embedding(init(a, b))
end

function (m::Embedding)(x::AutoGrad.Rec)
    m.mat = track(m.mat, x)
    m.mat[x.value, :]
end

function (m::Embedding)(x)
    m.mat = track(m.mat, x)
    m.mat[x, :]
end
