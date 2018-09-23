using BenchmarkTools,Knet

const N=1000

const sizes = (1,10,100,128,512,1000,1024,2048)

function f01(a,b)
    c=similar(a)
    n=length(a)
    for i=1:N; ccall(("add_32_01",Knet.libknet8),Nothing,(Cint,Float32,Ptr{Float32},Ptr{Float32}),n,b,a,c); end
    Knet.cudaDeviceSynchronize()
end

function f12(x,y)

    (dz,sx,nx,sy,ny,xlast,ylast,xdims,ydims,multi) = Knet.vbroadcast_shape(x,y)
    z = similar(x,dz)
    nz = length(z)
    for i=1:N; ccall(("add_32_12",Knet.libknet8),Nothing,(Cint,Ptr{Float32},Cint,Cint,Ptr{Float32},Cint,Cint,Ptr{Float32}),nz,x,sx,nx,y,sy,ny,z); end
    Knet.cudaDeviceSynchronize()
end

# other than first dim broadcast
# y is vector to be broadcasted, then ylast is broadcasted dim
function f13_x_y(x,y)
    (dz,sx,nx,sy,ny,xlast,ylast,xdims,ydims,multi) = Knet.vbroadcast_shape(x,y)
    z = similar(x,dz)
    brdcastdimstride = strides(x)[ylast]
    # if broadcast last dimension, nextstride is zero
    brdcastnextstride = ((ylast+1) > ndims(x) ? 0 : strides(x)[ylast+1])
    multidimsize = prod(size(x)[ylast+1:end])
    for i=1:N; ccall(("add_32_13_x_y",Knet.libknet8),Nothing,(Ptr{Float32},Ptr{Float32},Ptr{Float32},Cint,Cint,Cint,Cint,Cint),x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,length(x),length(y)) end
    Knet.cudaDeviceSynchronize()
end



#x is N-dim y is vector
function f14_x_y(x,y)
    (dz,sx,nx,sy,ny,xlast,ylast,xdims,ydims,multi) = Knet.vbroadcast_shape(x,y)
    z = similar(x,dz)
    flat_dimsize=(length(x)/length(y))
    for i=1:N; ccall(("add_32_14_x_y",Knet.libknet8),Nothing,(Ptr{Float32},Ptr{Float32},Ptr{Float32},Cint,Cint,Cint),x,y,z,length(y),length(x),flat_dimsize) end
    Knet.cudaDeviceSynchronize()
end


#r=1 f13_x_y, r=2 f14
# m=nrows n=ncols
for r in (0,1,2)
    println(r==0 ? "a[m,n].+b" : r==1 ? "a[m,n].+b[1,n]" : r==2 ? "a[m,n].+b[m,1]" : error())
    for s in sizes; print("\t$s"); end; println()
    for nrows in sizes
        print(nrows)
        for ncols in sizes
            a = KnetArray(rand(Float32,nrows,ncols))
            b = (r==0 ? rand(Float32) : r==1 ? KnetArray(rand(Float32,1,ncols)) : KnetArray(rand(Float32,nrows,1)))
            # bm = (r==0 ? (@benchmark f01($a,$b) seconds=1) : (@benchmark f12($a,$b) seconds=1))
            yvectorFlag= (nrows>=2048 && (100<=ncols<128 )) || (nrows>=512 && (128<=ncols<512 )) || (nrows>=100 && (512<=ncols ))
            if r==0
              bm=(@benchmark f01($a,$b) seconds=1)
            elseif (r==1 && (ncols<704 || nrows<512)) || (r==2)
            # elseif (r==1 && (ncols<704 || nrows<512)) || (r==2 && (!yvectorFlag))
              bm=(@benchmark f12($a,$b) seconds=1)
            elseif r==1
              bm=(@benchmark f13_x_y($a,$b) seconds=1)
            # else
            #   bm=(@benchmark f14_x_y($a,$b) seconds=1)
            end
            m = (round(Int, minimum(bm.times)/N))
            # 
            # m = (ncols*nrows*4)/(round(Int, minimum(bm.times)/N))
            print("\t$m")
            a=b=nothing; Knet.gc()
        end
        println()
    end
end

# COMMIT 3d32e16 2017-05-17
#
# a[m,n].+b
# 	1	10	100	128	512	1000	1024	2048
# 1	4264	4287	4295	4292	4302	4305	4306	4337
# 10	4288	4293	4305	4315	4365	4444	4459	4681
# 100	4295	4305	4457	4469	5209	6192	6238	11079
# 128	4299	4314	4472	4560	5600	6701	6787	17908
# 512	4303	4375	5211	5600	17957	30842	31749	59719
# 1000	4313	4436	6201	6699	30841	57163	58332	113552
# 1024	4312	4459	6245	6797	31668	58113	59591	116099
# 2048	4341	4683	11695	18025	59673	113487	116187	232430
# a[m,n].+b[1,n]
# 	1	10	100	128	512	1000	1024	2048
# 1	4304	4308	4317	4316	4330	4356	4356	4361
# 10	4307	4487	4576	4576	4706	5018	5104	5523
# 100	4315	4575	5006	5204	7389	10042	10129	16564
# 128	4316	4576	5202	5317	8237	11773	12162	22033
# 512	4325	4707	7414	8240	21068	33924	34016	60985
# 1000	4357	5006	10047	11783	36528	72782	73926	137811
# 1024	4355	5100	10153	12170	38988	61619	62400	115017
# 2048	4361	5531	16586	20558	76857	118232	119759	227061
# a[m,n].+b[m,1]
# 	1	10	100	128	512	1000	1024	2048
# 1	4311	4311	4316	4316	4325	4355	4354	4359
# 10	4307	4314	4366	4371	4696	5065	5075	5873
# 100	4318	4406	4897	5040	7304	10312	10450	17811
# 128	4318	4366	4861	5018	7302	10195	10441	18789
# 512	4334	4484	6512	7154	18362	31899	32853	61838
# 1000	4356	4729	8474	9675	32089	59411	60838	119014
# 1024	4356	4732	8459	9856	32833	60652	62001	121412
# 2048	4356	5196	13393	18371	61866	118420	121254	242820

# COMMIT: ccad0cb 2017-04-15
# a[m,n].+b
# 	1	10	100	1000
# 1	4264	4272	4281	4299
# 10	4268	4277	4298	4424
# 100	4275	4299	4422	6173
# 1000	4300	4412	6173	57050
# a[m,n].+b[1,n]
# 	1	10	100	1000
# 1	4296	4295	4312	4354
# 10	4304	4481	4573	5001
# 100	4311	4573	4992	10034
# 1000	4345	4998	10047	69640
# a[m,n].+b[m,1]
# 	1	10	100	1000
# 1	4304	4306	4314	4349
# 10	4300	4313	4359	5050
# 100	4314	4402	4865	10295
# 1000	4354	4718	8433	59328
