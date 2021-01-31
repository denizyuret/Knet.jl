export Residual


struct Residual
    f1
    f2
    activation
    Residual(f1, f2=identity; activation=nothing) = new(f1, f2, activation)
end


function (r::Residual)(x)
    y = r.f1(x) + r.f2(x)
    r.activation === nothing ? y : r.activation.(y)
end

