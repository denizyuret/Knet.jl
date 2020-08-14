using Knet, Test
macro timeinclude(x)
    :(print($x); print("\t"); @time include($x))
end

@testset "Knet" begin

#                            commit 686 ee1 c6a 0b5 e3a 9.2 8.3 8.3 6cb 6cb 8.6 6cb 6cb
#                           machine rzr  ci rzr  sc  sc tig ai5 ai4 tr5 tr4 aws osx os4
@timeinclude("kptr.jl")          #   2   ?   4   3   8  16   1   1   0   0  20   0   0
@timeinclude("gpu.jl")           #   2   3   1   2   2   6   1   1   0   0   2   0   0
@timeinclude("gcnode.jl")        #   1
@timeinclude("ops20.jl")         #
@timeinclude("distributions.jl") ##   1   1   1   1   1   2   1   1   2   1   3   3   2
@timeinclude("dropout.jl")       #   3   6   1   2   8   5                   2
@timeinclude("statistics.jl")    #   7
@timeinclude("bmm.jl")           #   9  10   3   9 
@timeinclude("linalg.jl")        #  17  33  17  22  22  62  24  14  22   7  28  33  19
@timeinclude("loss.jl")          #  17  32  13  19  20  10                   4
@timeinclude("cuarray.jl")       #  18  25
@timeinclude("batchnorm.jl")     #  19  32  14  23  22  93
@timeinclude("serialize.jl")     #  20  16   2  11   1
@timeinclude("jld.jl")           #   2  11  11   9  26
@timeinclude("karray.jl")        #  27  49  20  27  21  55  19  12   -   -  21   -   0
@timeinclude("reduction.jl")     #  42  67  37  52  55 106  40  21  29  11  57  55  29
@timeinclude("rnn.jl")           #  43  95  52  54  41  81                  12
@timeinclude("conv.jl")          #  47  71  33  48  51 107  22  12  62  47  26  44  16
@timeinclude("update.jl")        ##  64 316  61  82  60  61  29  26 100  22  72  25  23
@timeinclude("unary.jl")         #  76 172  86 103 103 122  42   6  36   4  56  67  11
@timeinclude("binary.jl")        #  89 124  64  75  80  56  34  19 491 119  51  53  25
#TODO include("data.jl")
#TODO include("hyperopt.jl")
#TODO include("progress.jl")
#TODO include("random.jl")
#TODO include("train.jl")
#TODO include("uva.jl")

end
