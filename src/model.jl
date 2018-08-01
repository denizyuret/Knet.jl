export track, back!, getgrad, setgrad!, params

abstract type Model end

"""
    tracked_x = track(x)

track an input, `x` can be tuple, array or dict. If x is already tracked, it will be replaced with a new tape.

    tracked_p = track(p, x)

track p on the same tape of x, return tracked p. If x is not tracked, it will also untrack p.
"""
track(x::AutoGrad.Rec) = track(x.value)
track(x, tape::AutoGrad.Tape=AutoGrad.Tape()) = AutoGrad.Rec(x, tape)
track(x, y) = x
track(x, y::AutoGrad.Rec) = AutoGrad.Rec(x, y.tapes[])
track(x::AutoGrad.Rec, y) = x.value
track(x::AutoGrad.Rec, y::AutoGrad.Rec) = x in y.tapes[] ? x : track(x.value, y)

"""
run backward pass of a tracked output, returns the gradient of input.

example:

```
x = track([1,-1,1])
p = track([2,3,4], x)
y = x .* p
back!(y, [1,1,1]) // =>[2,3,4]
getgrad(p) // =>[1,-1,1]
```
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
Get the gradient of a tracked variable. Return nothing if `x` is not used.
"""
function getgrad(x::AutoGrad.Rec)
    x.nodes[].outgrad
end

"""
Set the gradient of a tracked variable.
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


export Affine, Chain, LSTM, Embedding, RNN, GRU

mutable struct Affine <: Model
    weight
    bias
end

Affine(a::Integer, b::Integer; init=xavier) = Affine(init(a, b), zeros(1, b))

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

mutable struct RNN <: Model
    weight
    bias
end

function RNN(a::Integer, b::Integer; init=rand)
    RNN(init(a+b, b), zeros(1, b))
end

function (m::RNN)(x, h)
    m.weight = track(m.weight, x)
    m.bias   = track(m.bias, x)

    tanh([x h] * m.weight .+ m.bias)
end

mutable struct GRU <: Model
    Wih
    Whh
    bih
    bhh
end

function GRU(a::Integer, b::Integer; init=rand)
    GRU(init(a, 3b), init(b, 3b), zeros(1, 3b), zeros(1, 3b))
end

function (m::GRU)(x, h)
    m.Wih = track(m.Wih, x)
    m.Whh = track(m.Whh, x)
    m.bih = track(m.bih, x)
    m.bhh = track(m.bhh, x)

    gi = x * m.Wih .+ m.bih
    gh = h * m.Whh .+ m.bhh

    hsize = size(h, 2)

    ir, ii, in = gi[:, 1:hsize], gi[:, 1+hsize:2hsize], gi[:, 1+2hsize:3hsize]
    hr, hi, hn = gh[:, 1:hsize], gh[:, 1+hsize:2hsize], gh[:, 1+2hsize:3hsize]

    rgate = sigm(ir + hr)
    igate = sigm(ii + hi)
    ngate = tanh(in + rgate .* hn)
    ngate + igate .* (h - ngate)
end