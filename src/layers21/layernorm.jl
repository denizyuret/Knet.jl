export LayerNorm
using Statistics
using Knet.Train20: param

# TODO: this is slow, need a kernel, maybe https://github.com/tensorflow/tensorflow/pull/6205/files
# TODO: support other axes

"""
    LayerNorm(size::Integer; eps=1e-5)
    LayerNorm(γ, β, ϵ)

References:
* [Ba, Kiros and Hinton 2016](https://arxiv.org/abs/1607.06450) Layer Normalization
* torch.nn.LayerNorm
* tf.keras.layers.LayerNormalization
"""
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
