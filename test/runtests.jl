#                            commit 8.3 8.3 6cb 6cb 8.2 6cb 6cb
#                           machine ai5 ai4 tr5 tr4 aws osx os4
#@time include("kptr.jl")         #   1   1   0   0   2   0   0
@time include("gpu.jl")           #   1   1   0   0  13   0   0  8 pass
@time include("distributions.jl") #   1   1   2   1   2   3   2  3 pass
@time include("update.jl")        #  29  26 100  22 103  25  23  9 pass, 9 error
@time include("karray.jl")        #  19  12   -   -  16   -   0  107 pass, 3 fail, 60 error
@time include("linalg.jl")        #  24  14  22   7  19  33  19  1 pass, 1 error
@time include("conv.jl")          #  22  12  62  47  34  44  16  0 pass
@time include("broadcast.jl")     #  34  19 491 119  32  53  25  3 pass, 3 error ??
@time include("unary.jl")         #  42   6  36   4  44  67  11  3 pass, 1 error ??
@time include("reduction.jl")     #  40  21  29  11  32  55  29  3 pass, 1 error
