macro cuda(lib,fun,x...)
    if Libdl.find_library(["lib$lib"], []) == ""
        msg = "Cannot find lib$lib, please install it and rerun Pkg.build(\"Knet\")."
        :(error($msg))
    else
        f2 = ("$fun","lib$lib")
        fx = Expr(:ccall, f2, :UInt32, x...)
        err = "$lib.$fun error "
        quote
            local _r = $fx
            if _r != 0
                warn($err, _r)
                Base.show_backtrace(STDOUT, backtrace())
            end
        end
    end
end

let GPU=-1, handles=Dict()
    global gpu, cublashandle, cudnnhandle

    gpu()=GPU  # (d=Cint[-1];@cuda(cudart,cudaGetDevice,(Ptr{Cint},),d);d[1])

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

"""
gpu() returns the id of the active GPU device or -1 if none are active.

gpu(d::Int) activates the GPU device d if 0 <= d < gpucount().

gpu(true) activates the GPU device with the most available memory.

gpu(false) deactivates GPU devices.    
""" gpu

gpucount()=(try; p=Cint[0]; eval(:(ccall(("cudaGetDeviceCount","libcudart"),UInt32,(Ptr{Cint},),$p))); p[1]; catch; 0; end)
gpumem()=(f=Csize_t[0];m=Csize_t[0]; @cuda(cudart,cudaMemGetInfo,(Ptr{Csize_t},Ptr{Csize_t}),f,m); (Int(f[1]),Int(m[1])))
gpufree()=gpumem()[1]
gpuinfo(msg="")=(print("$msg "); println((gpumem()...,meminfo()...)))
gpusync()=@cuda(cudart,cudaDeviceSynchronize,())

typealias Cptr Ptr{Void}

function cublasCreate()
    handleP = Cptr[0]
    @cuda(cublas,cublasCreate_v2, (Ptr{Cptr},), handleP)
    handle = handleP[1]
    atexit(()->@cuda(cublas,cublasDestroy_v2, (Cptr,), handle))
    return handle
end

function cudnnCreate()
    handleP = Cptr[0]
    @cuda(cudnn,cudnnCreate,(Ptr{Cptr},), handleP)
    handle = handleP[1]
    atexit(()->@cuda(cudnn,cudnnDestroy,(Cptr,), handle))
    return handle
end

