# const A = B works in all julia versions as typealias replacement but
# const A{T} = B{T} only works after 0.6.
# This does the right thing without causing warnings
macro typealias6(t1,t2)
    if VERSION >= v"0.6-"
        esc(:(const $t1 = $t2))
    else
        Expr(:typealias, t1, t2)
    end
end

# @compat has issues, using the _dot functions from AutoGrad that work with Julia 4,5,6:

using AutoGrad: exp_dot, log_dot, sqrt_dot, abs_dot, abs2_dot, sign_dot, tanh_dot

if VERSION >= v"0.6-"
    @eval relu_dot(x) = relu.(x)
    @eval sigm_dot(x) = sigm.(x)
else
    @eval relu_dot(x) = relu(x)
    @eval sigm_dot(x) = sigm(x)
end
