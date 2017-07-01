# type BatchMoments
#     μ
#     σ
#     momentum::Real
# end
#
# BatchMoments(; momentum=0.9) = BatchMoments(0., 1., momentum)
#
# function Base.push!(b::BatchMoments, μ, σ)
#     b.μ = b.momentum .* b.μ .+ (1 - b.momentum) .* b.μ
#     b.σ = b.momentum .* b.σ .+ (1 - b.momentum) .* b.σ
# end
#
# getmoments(bm::BatchMoments) = (bm.μ, bm.σ)
#

type BatchMoments
    μs::Vector
    σs::Vector
    count::Int
end

BatchMoments(n::Integer) = BatchMoments(Any[0 for _=1:n], Any[1 for _=1:n], 0)

function Base.push!(ms::BatchMoments, μ, σ)
    n = length(ms.μs)
    ms.count = ms.count == n ? 1 : ms.count + 1
    ms.μs[ms.count] = μ
    ms.σs[ms.count] = σ
end

getmoments(ms::BatchMoments) = (mean(ms.μs), mean(ms.σs))

import AutoGrad
Base.size(a::AutoGrad.Rec, d1::Integer, d2::Integer, dx::Vararg{Integer}) = size(getval(a), d1, d2, dx...)

# Batch Normalization Layer
# works both for convolutional and fully connected layers
function batchnorm(w, x, bmom::BatchMoments; mode=:train, ϵ=1e-5)
    if mode == :train
        nd = ndims(x)
        # eg. d = (2,) for fc layers and d=(1,2,4) for conv layers
        d = tuple((1:nd-2)..., nd)

        # μ = mean(x, d)      # not supported by AutoGrad
        # σ = sqrt(ϵ .+ varm(x, μ, d)) # not supported by AutoGrad

        s = prod(size(x, d...))
        μ = sum(x, d) ./ s
        σ = sqrt(ϵ + sum((x .- μ).^2, d) ./ s)

        # we need getval in backpropagation
        push!(bmom, getval(μ), getval(σ))
    elseif mode == :test
        μ, σ = getmoments(bmom)
    end
    return w[1] .* (x .- μ) ./ σ .+ w[2]
end
