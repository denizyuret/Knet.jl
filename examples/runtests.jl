using Knet, Base.Test
isapprox3(a,b;o...)=all(map((x,y)->(x==y || isapprox(x,y;o...)), a,b))
load_only = true

include("linreg.jl")
@time @show test1 = linreg()
@test test1 == (0.0005497372347062403,32.77256166946497,0.1124434940652303)
# 8.121040 seconds (8.80 M allocations: 473.322 MB, 1.58% gc time)

include("mnist2d.jl")
@time @show test2 = mnist2d()
@test test2 == (0.10628127f0,24.865437f0,3.5134742f0)
# 9.759260 seconds (9.98 M allocations: 690.400 MB, 1.90% gc time)

include("mnist4d.jl")
@time @show test3 = mnist4d()
@test isapprox(test3[1], 0.08003744f0; rtol=.01)
@test isapprox(test3[2], 19.31503f0;   rtol=.01)
@test isapprox(test3[3], 8.413661f0;   rtol=.1)
# 20.173298 seconds (14.43 M allocations: 897.290 MB, 1.11% gc time)

include("mnistpixels.jl")
@time @show test4 = mnistpixels()
@test test4 == (0.1216,2.3023174f0,10.4108f0,30.598776f0)
# 12.237608 seconds (45.53 M allocations: 1.201 GB, 2.45% gc time)

include("mnistsparse.jl")
@time @show test5 = mnistsparse()
@test test5[1] == (adense,adense,0.10628127f0,24.865437f0,3.5134742f0,0.100041345f0,0.9681833333333333,0.114785746f0,0.9641)
@test test5[2] == (adense,sparse,0.1062698f0,24.866688f0,3.5134742f0,0.100718f0,0.9679333333333333,0.1149149f0,0.9642)
@test isapprox3(test5[3], (sparse,adense,0.106281266f0,24.865438f0,3.5134728f0,0.1000414f0,0.9681833333333333,0.11478577f0,0.9641); rtol=.02)
@test isapprox3(test5[4], (sparse,sparse,0.10627592f0,24.866186f0,3.5134728f0,0.09992511f0,0.9681666666666666,0.11481799f0,0.9641); rtol=.02)
# 48.989648 seconds (32.63 M allocations: 2.777 GB, 1.55% gc time)

include("adding.jl")
@time @show test6 = adding()
@test test6 == (0.048857126f0,5.6036315f0,3.805253f0)
# 15.512905 seconds (20.54 M allocations: 999.662 MB, 2.18% gc time)

include("rnnlm.jl")
@time @show test7 = rnnlm("ptb.valid.txt ptb.test.txt")
@test isapprox3(test7, (814.9780887272417,541.2457922913605,267.626257438979,120.16170771885587); rtol=.02)
# 33.872548 seconds (22.48 M allocations: 2.218 GB, 1.39% gc time)
