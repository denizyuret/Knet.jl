#                            commit 0b5 e3a 9.2 8.3 8.3 6cb 6cb 8.6 6cb 6cb
#                           machine  sc  sc tig ai5 ai4 tr5 tr4 aws osx os4
@time include("kptr.jl")          #   3   8  16   1   1   0   0  20   0   0
@time include("gpu.jl")           #   2   2   6   1   1   0   0   2   0   0
@time include("serialize.jl")     #  11   1
@time include("distributions.jl") #   1   1   2   1   1   2   1   3   3   2
@time include("bmm.jl")           #   9 
@time include("dropout.jl")       #   2   8   5                   2
@time include("loss.jl")          #  19  20  10                   4
@time include("karray.jl")        #  27  21  55  19  12   -   -  21   -   0
@time include("batchnorm.jl")     #  23  22  93
@time include("linalg.jl")        #  22  22  62  24  14  22   7  28  33  19
@time include("jld.jl")           #   9  26
@time include("rnn.jl")           #  54  41  81                  12
@time include("conv.jl")          #  48  51 107  22  12  62  47  26  44  16
@time include("reduction.jl")     #  52  55 106  40  21  29  11  57  55  29
@time include("update.jl")        #  82  60  61  29  26 100  22  72  25  23
@time include("broadcast.jl")     #  75  80  56  34  19 491 119  51  53  25
@time include("unary.jl")         # 103 103 122  42   6  36   4  56  67  11
