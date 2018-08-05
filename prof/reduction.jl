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
            GC.gc();Knet.knetgc();GC.gc()
        end
        println()
    end
end

# COMMIT 3d32e16 2017-05-17 merge
#
# sum(a,0)
# 	1	10	100	1000
# 1	31804	31817	31814	31822
# 10	31790	31852	31827	32158
# 100	31848	31876	32085	35014
# 1000	31831	32072	35023	65961
# sum(a,1)
# 	1	10	100	1000
# 1	4689	4689	4931	13799
# 10	4936	4938	5436	16528
# 100	4936	4938	5470	17198
# 1000	6457	6608	7302	48299
# sum(a,2)
# 	1	10	100	1000
# 1	4690	4689	4691	6779
# 10	4690	4748	4885	8096
# 100	4935	5208	6864	26480
# 1000	13809	15307	32166	216076


# COMMIT: ccad0cb 2017-04-15 (added general reductions, same as last)
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

# COMMIT: c78d436 2017-04-04 (fixed large array bug, but slowed last col sum(a,1) and last row sum(a,2))
# sum(a,0)
# 	1	10	100	1000
# 1	32333	32391	32305	32404
# 10	32388	32403	32440	32703
# 100	32411	32407	32668	35571
# 1000	32362	32618	35586	66601
# sum(a,1)
# 	1	10	100	1000
# 1	4690	4690	4924	13812
# 10	4939	4941	5427	16532
# 100	4932	4938	5470	17179
# 1000	6506	6583	7299	48242
# sum(a,2)
# 	1	10	100	1000
# 1	4690	4691	4691	6773
# 10	4690	4750	4884	8085
# 100	4929	5197	6866	26475
# 1000	13841	15303	32141	216153


# COMMIT: 89c2026 2017-04-02 (buggy with large arrays)
# sum(a,0)
# 	1	10	100	1000
# 1	32326	32417	32342	32467
# 10	32390	32346	32232	32597
# 100	32212	32345	32571	35466
# 1000	32363	32547	35519	66645
# sum(a,1)
# 	1	10	100	1000
# 1	4470	4528	4777	10954
# 10	4713	4777	5142	12531
# 100	4699	4778	5200	12915
# 1000	6245	6435	7126	42014
# sum(a,2)
# 	1	10	100	1000
# 1	4480	4481	4483	6546
# 10	4541	4601	4695	7929
# 100	4777	4975	6781	26433
# 1000	10970	12016	28019	208529
