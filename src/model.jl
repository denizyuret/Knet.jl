export Model, track, back!, getgrad, setgrad!

"""

"""
abstract type Model end

"""

"""
track(x::AutoGrad.Rec) = track(x.value)
track(x, tape::AutoGrad.Tape=AutoGrad.Tape()) = AutoGrad.Rec(x, tape)
track(x, y::AutoGrad.Rec) = AutoGrad.Rec(x, y.tapes[])
track(x::AutoGrad.Rec, y) = x.value
track(x::AutoGrad.Rec, y::AutoGrad.Rec) = track(x.value, y)

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
function params(m::Model)
    set = ObjectIdDict()

    traverse(m::AutoGrad.Rec) = set[m] = true
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


export Affine, MLP

mutable struct Affine <: Model
    W
    b
end

Affine(a::Integer, b::Integer; init=xavier) = Affine(init(b, a), init(b))

function (m::Affine)(x)
    m.W = track(m.W, x)
    m.b = track(m.b, x)
    m.W * x .+ m.b
end

mutable struct Chain <: Model
    layers::Vector
end

Chain(x...) = Chain(collect(x))

function (m::Chain)(x)
    foldl((m, x) -> m(x), x, m.layers)
end

mutable struct LSTM <: Model
    weight
    bias
end

function LSTM(a::Integer, b::Integer; init=rand)
    LSTM(init(a+b, 4b), init(1, 4b))
end

function (m::LSTM)(x, hidden, cell)
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
