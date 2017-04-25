using BenchmarkTools,Knet

const N=1000

const sizes = (1,10,100,128,512,1000,1024,2048)

function f01(a,b)
    c=similar(a)
    n=length(a)
    for i=1:N; ccall(("add_32_01",Knet.libknet8),Void,(Cint,Float32,Ptr{Float32},Ptr{Float32}),n,b,a,c); end
    Knet.cudaDeviceSynchronize()
end

function f12(x,y)

    (dz,sx,nx,sy,ny,xlast,ylast,xdims,ydims,multi) = Knet.vbroadcast_shape(x,y)
    z = similar(x,dz)
    nz = length(z)
    for i=1:N; ccall(("add_32_12",Knet.libknet8),Void,(Cint,Ptr{Float32},Cint,Cint,Ptr{Float32},Cint,Cint,Ptr{Float32}),nz,x,sx,nx,y,sy,ny,z); end
    Knet.cudaDeviceSynchronize()
end

# other than first dim broadcast
# y is vector to be broadcasted, then ylast is broadcasted dim
function f13_x_y(x,y)
    (dz,sx,nx,sy,ny,xlast,ylast,xdims,ydims,multi) = Knet.vbroadcast_shape(x,y)
    z = similar(x,dz)
    brdcastdimstride = strides(x)[ylast]
    # if broadcast last dimension, nextstride is zero
    brdcastnextstride = ((ylast+1) > ndims(x) ? 0: strides(x)[ylast+1])
    multidimsize = prod(size(x)[ylast+1:end])
    for i=1:N; ccall(("add_32_13_x_y",Knet.libknet8),Void,(Ptr{Float32},Ptr{Float32},Ptr{Float32},Cint,Cint,Cint,Cint,Cint),x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,length(x),length(y)) end
    Knet.cudaDeviceSynchronize()
end

# #y is matrix x is vector
# function f14_x_y(y,x)
#     for i=1:N; ccall(("add_32_14_y_x",Knet.libknet8),Void,(Ptr{Float32},Ptr{Float32},Ptr{Float32},Cint,Cint),y,x,z,length(x),length(y)) end
#     Knet.cudaDeviceSynchronize()
# end

#x is matrix y is vector
function f14_x_y(x,y)
    (dz,sx,nx,sy,ny,xlast,ylast,xdims,ydims,multi) = Knet.vbroadcast_shape(x,y)
    z = similar(x,dz)
    flat_dimsize=(length(x)/length(y))
    for i=1:N; ccall(("add_32_14_x_y",Knet.libknet8),Void,(Ptr{Float32},Ptr{Float32},Ptr{Float32},Cint,Cint,Cint),x,y,z,length(y),length(x),flat_dimsize) end
    Knet.cudaDeviceSynchronize()
end


#r=1 f13_x_y, r=2 f14
# for r in (0,1,2)
# m=nrows n=ncols
for r in (2)
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
            elseif (r==1 && (ncols<704 || nrows<512)) || (r==2 && (!yvectorFlag))
              bm=(@benchmark f12($a,$b) seconds=1)
            elseif r==1
              bm=(@benchmark f13_x_y($a,$b) seconds=1)
            else
              bm=(@benchmark f14_x_y($a,$b) seconds=1)
            end

            # bm = (r==0 ? (@benchmark f01($a,$b) seconds=1) : ((r==1 && (ncols<704 || nrows<512)) || (r==2 && (!yvectorFlag))) ? (@benchmark f12($a,$b) seconds=1) : r==1 ? (@benchmark f13_x_y($a,$b) seconds=1) : (@benchmark f14_x_y($a,$b) seconds=1) )
            m = (round(Int, minimum(bm.times)/N))
            # m = (ncols*nrows*4)/(round(Int, minimum(bm.times)/N))
            print("\t$m")
            a=b=nothing; gc(); Knet.knetgc(); gc()
        end
        println()
    end
end


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
