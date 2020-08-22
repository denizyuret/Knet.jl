export LayerNorm
using Statistics

## LayerNorm: https://arxiv.org/abs/1607.06450: Layer Normalization
# TODO: this is slow, need a kernel, maybe https://github.com/tensorflow/tensorflow/pull/6205/files

struct LayerNorm; γ; β; ϵ; end

function LayerNorm(dmodel; eps=1e-5)
    γ = param(dmodel; init=ones)
    β = param(dmodel; init=zeros)
    LayerNorm(γ, β, eps)
end

function (l::LayerNorm)(x, o...)
    μ = mean(x,dims=1)
    σ = std(x,mean=μ,dims=1)
    ϵ = eltype(x)(l.ϵ)
    l.γ .* (x .- μ) ./ (σ .+ ϵ) .+ l.β # TODO: doing x .- μ twice?
end

# function (l::LayerNorm)(x::MaskedArray, o...)
#     MaskedArray(l(x.array), x.mask) # TODO: shouldn't normalization ignore masked values?
# end
