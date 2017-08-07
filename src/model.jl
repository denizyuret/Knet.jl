export Model, track, back!, getgrad, setgrad!

"""

"""
abstract type Model end

"""

"""
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


export Affine

mutable struct Affine <: Model
    W
    b
end

Affine(a::Integer, b::Integer, init=rand) = Affine(rand(b, a), rand(b))

function (m::Affine)(x)
    m.W = track(m.W, x)
    m.b = track(m.b, x)
    m.W * x .+ m.b
end
