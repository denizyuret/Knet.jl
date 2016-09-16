macro cuda(lib,fun,x...)
    if Libdl.find_library(["lib$lib"], []) != ""
        f2 = ("$fun","lib$lib")
        fx = Expr(:ccall, f2, :UInt32, x...)
        err = "$lib.$fun error "
        quote
            local _r = $fx
            if _r != 0
                warn($err * _r)
                Base.show_backtrace(STDOUT, backtrace())
            end
        end
    end
end

let GPU=-1, handles=Dict()
    global gpu, cublashandle, cudnnhandle

    "Return the active gpu device, or -1 for cpu."
    gpu()=GPU  # (d=Cint[-1];@cuda(cudart,cudaGetDevice,(Ptr{Cint},),d);d[1])

    "Use gpu with device id i for 0<=i<gpucount(), otherwise use cpu."
    function gpu(i::Int)
        if 0 <= i < gpucount()
            @cuda(cudart,cudaSetDevice, (Cint,), i)
            cublashandle = get!(cublasCreate, handles, (:cublas,i))
            cudnnhandle  = get!(cudnnCreate, handles, (:cudnn,i))
        else
            i = -1
            cublashandle = cudnnhandle = nothing
        end
        return (GPU = i)
    end

    "Pick the gpu with the most available memory if b=true, otherwise use cpu."
    function gpu(b::Bool)
        if b
            pick = mem = -1
            for i=0:gpucount()-1
                @cuda(cudart,cudaSetDevice, (Cint,), i)
                imem = gpufree()
                if imem > mem
                    pick = i
                    mem = imem
                end
            end
            gpu(pick)
        else
            gpu(-1)
        end
    end
end

gpucount()=(p=Cint[0]; @cuda(cudart,cudaGetDeviceCount,(Ptr{Cint},),p); p[1])
gpumem()=(f=Csize_t[0];m=Csize_t[0]; @cuda(cudart,cudaMemGetInfo,(Ptr{Csize_t},Ptr{Csize_t}),f,m); (Int(f[1]),Int(m[1])))
gpufree()=gpumem()[1]
gpuinfo(msg="")=(print("$msg "); println((gpumem()...,meminfo()...)))

function cublasCreate()
    handleP = Ptr{Void}[0]
    @cuda(cublas,cublasCreate_v2, (Ptr{Ptr{Void}},), handleP)
    handle = handleP[1]
    atexit(()->@cuda(cublas,cublasDestroy_v2, (Ptr{Void},), handle))
    return handle
end

function cudnnCreate()
    handleP = Ptr{Void}[0]
    @cuda(cudnn,cudnnCreate,(Ptr{Ptr{Void}},), handleP)
    handle = handleP[1]
    atexit(()->@cuda(cudnn,cudnnDestroy,(Ptr{Void},), handle))
    return handle
end

