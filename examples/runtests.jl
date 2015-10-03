using Knet, Base.Test
isapprox3(a,b;o...)=all(map((x,y)->(x==y || isapprox(x,y;o...)), a,b))
load_only = true

include("linreg.jl")
@time @show test1 = linreg()
@test test1 == (0.0005497372347062403,32.77256166946497,0.1124434940652303)
# 7.614517 seconds (8.15 M allocations: 420.915 MB, 1.58% gc time)
# after compile: 1.640862 seconds (939.00 k allocations: 120.314 MB, 2.22% gc time)

include("mnist2d.jl")
@time @show test2 = mnist2d()
@test test2 == (0.10628127f0,24.865437f0,3.5134742f0)
#  8.911226 seconds (8.82 M allocations: 583.053 MB, 1.86% gc time)
# after compile: 5.703016 seconds (6.19 M allocations: 331.939 MB, 1.95% gc time)

include("mnist4d.jl")
@time @show test3 = mnist4d()
@test isapprox(test3[1], 0.08003744f0; rtol=.01)
@test isapprox(test3[2], 19.31503f0;   rtol=.01)
@test isapprox(test3[3], 8.413661f0;   rtol=.1)
# 18.928054 seconds (12.91 M allocations: 797.344 MB, 1.08% gc time)

include("mnistsparse.jl")
@time @show test5 = mnistsparse()
@test test5[1] == (adense,adense,0.10628127f0,24.865437f0,3.5134742f0,0.100041345f0,0.9681833333333333,0.114785746f0,0.9641)
@test test5[2] == (adense,sparse,0.1062698f0,24.866688f0,3.5134742f0,0.100718f0,0.9679333333333333,0.1149149f0,0.9642)

@test isapprox(test5[3][3], 0.10628127f0; rtol=0.002)
@test isapprox(test5[3][4], 24.865437f0; rtol=0.0005)
@test isapprox(test5[3][5], 3.5134742f0; rtol=0.01)
@test isapprox(test5[3][6], 0.100041345f0; rtol=0.02)
@test isapprox(test5[3][7], 0.9681833333333333; rtol=0.001)
@test isapprox(test5[3][8], 0.114785746f0; rtol=0.02)
@test isapprox(test5[3][9], 0.9641; rtol=0.002)

@test isapprox(test5[4][3], 0.10628127f0; rtol=0.002)
@test isapprox(test5[4][4], 24.865437f0; rtol=0.0005)
@test isapprox(test5[4][5], 3.5134742f0; rtol=0.01)
@test isapprox(test5[4][6], 0.100041345f0; rtol=0.02)
@test isapprox(test5[4][7], 0.9681833333333333; rtol=0.001)
@test isapprox(test5[4][8], 0.114785746f0; rtol=0.02)
@test isapprox(test5[4][9], 0.9641; rtol=0.002)
#  40.934952 seconds (24.52 M allocations: 2.219 GB, 1.37% gc time)

include("mnistpixels.jl")
@time @show test4 = mnistpixels()
@test test4 == (0.1216,2.3023174f0,10.4108f0,30.598776f0)
#  11.723709 seconds (45.43 M allocations: 1.196 GB, 2.59% gc time)

include("adding.jl")
@time @show test6 = adding()
@test test6 == (0.048857126f0,5.6036315f0,3.805253f0)
# 12.480548 seconds (18.24 M allocations: 840.938 MB, 2.30% gc time)

include("rnnlm.jl")
@time @show test7 = rnnlm("ptb.valid.txt ptb.test.txt")
@test isapprox3(test7, (814.9780887272417,541.2457922913605,267.626257438979,120.16170771885587); rtol=.02)
# 33.411935 seconds (22.31 M allocations: 2.210 GB, 1.51% gc time)
