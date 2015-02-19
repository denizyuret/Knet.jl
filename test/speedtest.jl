using HDF5
using CUDArt
using KUnet
const libkunet = find_library(["libkunet"], ["."])
typealias Mat CudaArray{Float32,2}
typealias Cmat Ptr{Float32}

function forward!(x::Matrix{Float32}, x1::Mat, w1::Mat, b1::Mat, x2::Mat, w2::Mat, b2::Mat, x3::Mat, y::Matrix{Float32})
    batch = 937
    xrows,xcols = size(x)
    yrows,ycols = size(y)
    for i=1:batch:xcols
        copy!(x1,1,x,(i-1)*xrows+1,length(x1))
        CUBLAS.gemm!('N','N',1.0f0,w1,x1,0.0f0,x2)
        ccall((:badd,libkunet),Void,(Cint,Cint,Cmat,Cmat),size(x2,1),size(x2,2),x2,b1)
        ccall((:reluforw,libkunet),Void,(Cint,Cmat),length(x2),x2)
        CUBLAS.gemm!('N','N',1.0f0,w2,x2,0.0f0,x3)
        ccall((:badd,libkunet),Void,(Cint,Cint,Cmat,Cmat),size(x3,1),size(x3,2),x3,b2)
        copy!(y,(i-1)*yrows+1,x3,1,length(x3))
    end
    y
end

function speedtest()
    x = h5read("devx.h5","/data")
    l1 = Layer("dev1.h5")
    l2 = Layer("dev2.h5")
    l1.xforw = KUnet.noop
    l2.xforw = KUnet.noop
    l1.w = CudaArray(l1.w)
    l1.b = CudaArray(l1.b)
    l2.w = CudaArray(l2.w)
    l2.b = CudaArray(l2.b)
    xrows,xcols = size(x)
    yrows,ycols = size(l2.w,1), xcols
    batch = 937
    y = similar(x, yrows, ycols)
    # xx = similar(l1.w,(xrows,batch))
    xx = CudaArray(x[:,1:batch])
    l1.y = similar(l1.w,(size(l1.w,1), batch))
    l2.y = similar(l2.w,(size(l2.w,1), batch))
    @time forward!(x, xx, l1.w, l1.b, l1.y, l2.w, l2.b, l2.y, y)
end

function speedtest2()
    batch = 937
    x = h5read("devx.h5","/data")
    l1 = Layer("dev1.h5")
    l2 = Layer("dev2.h5")
    l1.xforw = KUnet.noop
    l2.xforw = KUnet.noop
    l1.w = CudaArray(l1.w)
    l1.b = CudaArray(l1.b)
    l2.w = CudaArray(l2.w)
    l2.b = CudaArray(l2.b)
    # xrows,xcols = size(x)
    # yrows,ycols = size(l2.w,1), xcols
    # y = similar(x, yrows, ycols)
    # xx = similar(l1.w,(xrows,batch))
    # xx = CudaArray(x[:,1:batch])
    # l1.y = similar(l1.w,(size(l1.w,1), batch))
    # l2.y = similar(l2.w,(size(l2.w,1), batch))
    net = [l1,l2]
    @time KUnet.predict(net, x, batch)
    @time KUnet.predict(net, x, batch)
end

function speedtest3()
    blas_set_num_threads(20)
    batch = 937
    x = h5read("devx.h5","/data")
    l1 = Layer("dev1.h5")
    l2 = Layer("dev2.h5")
    l1.xforw = KUnet.noop
    l2.xforw = KUnet.noop
    # l1.w = CudaArray(l1.w)
    # l1.b = CudaArray(l1.b)
    # l2.w = CudaArray(l2.w)
    # l2.b = CudaArray(l2.b)
    # xrows,xcols = size(x)
    # yrows,ycols = size(l2.w,1), xcols
    # y = similar(x, yrows, ycols)
    # xx = similar(l1.w,(xrows,batch))
    # xx = CudaArray(x[:,1:batch])
    # l1.y = similar(l1.w,(size(l1.w,1), batch))
    # l2.y = similar(l2.w,(size(l2.w,1), batch))
    net = [l1,l2]
    @time KUnet.predict(net, x, batch)
    @time KUnet.predict(net, x, batch)
end

function speedtest4()
    blas_set_num_threads(20)
    batch = 937
    x = h5read("devx.h5","/data")
    y = h5read("devy.h5","/data")
    l1 = KUnet.Layer("dev1.h5")
    l2 = KUnet.Layer("dev2.h5")
    l1.xforw = KUnet.noop
    l2.xforw = KUnet.noop
    net = [l1,l2]
    xx = x[:,1:batch]
    yy = y[:,1:batch]
    @time KUnet.backprop(net, xx, yy)
    @time KUnet.backprop(net, xx, yy)
    net
end

function speedtest5()
    blas_set_num_threads(20)
    batch = 937
    x = h5read("devx.h5","/data")
    y = h5read("devy.h5","/data")
    l1 = KUnet.Layer("dev1.h5")
    l2 = KUnet.Layer("dev2.h5")
    l1.xforw = KUnet.noop
    l2.xforw = KUnet.noop
    l1.w = CudaArray(l1.w)
    l1.b = CudaArray(l1.b)
    l2.w = CudaArray(l2.w)
    l2.b = CudaArray(l2.b)
    net = [l1,l2]
    xx = CudaArray(x[:,1:batch])
    yy = CudaArray(y[:,1:batch])
    @time KUnet.backprop(net, xx, yy)
    @time KUnet.backprop(net, xx, yy)
    net
end

function speedtest6()
    blas_set_num_threads(20)
    batch = 937
    x = h5read("devx.h5","/data")
    y = h5read("devy.h5","/data")
    l1 = KUnet.Layer("rnd1.h5")
    l2 = KUnet.Layer("rnd2.h5")
    net = [l1,l2]
    @time KUnet.train(net, x, y; batch=937, iters=1, l2reg=0.5f0)
    @time KUnet.train(net, x, y; batch=937, iters=1, l2reg=0.5f0)
    net
end

function speedtest7()
    batch = 937
    x = h5read("devx.h5","/data")
    y = h5read("devy.h5","/data")
    l1 = KUnet.Layer("rnd1.h5")
    l2 = KUnet.Layer("rnd2.h5")
    l1.w = CudaArray(l1.w)
    l1.b = CudaArray(l1.b)
    l2.w = CudaArray(l2.w)
    l2.b = CudaArray(l2.b)
    net = [l1,l2]
    @time KUnet.train(net, x, y; batch=937, iters=1, l2reg=0.5f0)
    @time KUnet.train(net, x, y; batch=937, iters=1, l2reg=0.5f0)
    net
end
