using BenchmarkTools,Knet

const N=1000

const sizes = (1,10,100,1000)

function f2(x,r)
    nx = length(x)
    if r==0
        for i=1:N; y=ccall(("sum_32_20",Knet.libknet8),Float32,(Cint,Ptr{Float32}),nx,x); end
    elseif r==1
        y = similar(x, 1, size(x,2)); ny = length(y); sy = stride(x,2)
        for i=1:N; ccall(("sum_32_21",Knet.libknet8),Void,(Cint,Ptr{Float32},Cint,Cint,Ptr{Float32}),nx,x,sy,ny,y); end
    elseif r==2
        y = similar(x, size(x,1), 1); ny = length(y); sy = stride(x,1)
        for i=1:N; ccall(("sum_32_21",Knet.libknet8),Void,(Cint,Ptr{Float32},Cint,Cint,Ptr{Float32}),nx,x,sy,ny,y); end
    else
        error("not supported")
    end
    Knet.cudaDeviceSynchronize()
    return y
end

for region in (0,1,2)
    println("sum(a,$region)")
    for s in sizes; print("\t$s"); end; println()
    for nrows in sizes
        print(nrows)
        for ncols in sizes
            a = KnetArray(rand(Float32,nrows,ncols))
            bm = @benchmark f2($a,$region) seconds=1
            m = round(Int, minimum(bm.times)/N)
            print("\t$m")
            gc();Knet.knetgc();gc()
        end
        println()
    end
end


# COMMIT: ccad0cb 2017-04-15
# sum(a,0)
# 	1	10	100	1000
# 1	31741	31853	31864	31853
# 10	31779	31804	31899	32097
# 100	31818	31895	32155	35172
# 1000	31871	32094	35185	66605
# sum(a,1)
# 	1	10	100	1000
# 1	4619	4689	4892	13813
# 10	4885	4905	5404	16503
# 100	4848	4914	5442	17207
# 1000	6390	6579	7300	48356
# sum(a,2)
# 	1	10	100	1000
# 1	4611	4621	4606	6692
# 10	4689	4731	4830	8054
# 100	4887	5104	6839	26471
# 1000	13841	15296	32159	217215
