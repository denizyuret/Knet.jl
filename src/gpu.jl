const libknet8  = Libdl.find_library(["libknet8"], [Pkg.dir("Knet/src")])

function __init__()
    try
        gpu(true)
        info("Using GPU $(gpu())")
    catch e
        warn("$e: using the CPU.")
        gpu(false)
    end
end

macro cudart(f,x...)
    fx = Expr(:ccall, :($f,"libcudart"), :UInt32, x...)
    quote
        local _r = $fx
        if _r != 0
            warn("CUDA error $_r triggered from:")
            Base.show_backtrace(STDOUT, backtrace())
        end
    end
end

let GPU=-1, handles=Dict()
    global gpu, cublashandle, cudnnhandle

    "Return the active gpu device, or -1 for cpu."
    gpu()=GPU  # (d=Cint[-1];@cudart(:cudaGetDevice,(Ptr{Cint},),d);d[1])

    "Use gpu with device id i for i>=0, otherwise use cpu."
    function gpu(i::Int)
        GPU = i
        if i >= 0
            @cudart(:cudaSetDevice, (Cint,), i)
            cublashandle = get!(cublasCreate, handles, (:cublas,i))
            cudnnhandle  = get!(cudnnCreate, handles, (:cudnn,i))
        else
            cublashandle = cudnnhandle = nothing
        end
        return GPU
    end

    "Pick the gpu with the most available if b=true, otherwise use cpu."
    function gpu(b::Bool)
        if b
            pick = mem = -1
            for i=0:gpucount()-1
                @cudart(:cudaSetDevice, (Cint,), i)
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

function gpucount()
    ptr=Cint[0]
    @cudart(:cudaGetDeviceCount,(Ptr{Cint},),ptr)
    return Int(ptr[1])
end

function gpumem()
    mfree=Csize_t[1]; mtotal=Csize_t[1]
    @cudart(:cudaMemGetInfo,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    (Int(mfree[1]),Int(mtotal[1]))
end

gpufree()=gpumem()[1]

function gpuinfo(msg="")
    print("$msg ")
    ptrs = isdefined(:CUDArt) ? (:cuda_ptrs,length(CUDArt.cuda_ptrs)) : ()
    println((gpumem()...,meminfo()...,ptrs...))
end

function cublasCreate()
    handleP = Ptr{Void}[0]
    ret = ccall((:cublasCreate_v2, "libcublas"), UInt32, (Ptr{Ptr{Void}},), handleP)
    ret==0 || error("Could not create cublasHandle: $ret")
    handle = handleP[1]
    atexit(()->ccall((:cublasDestroy_v2, "libcublas"), UInt32, (Ptr{Void},), handle))
    return handle
end

function cudnnCreate()
    handleP = Ptr{Void}[0]
    ret = ccall((:cudnnCreate, "libcudnn"), UInt32, (Ptr{Ptr{Void}},), handleP)
    ret==0 || error("Could not create cudnnHandle: $ret")
    handle = handleP[1]
    atexit(()->ccall((:cudnnDestroy, "libcudnn"), UInt32, (Ptr{Void},), handle))
    return handle
end

# function gpulibs()
#     libs = true
#     lpath = [Pkg.dir("Knet/src")]
#     for l in ("libknet8", "libcuda", "libcudart", "libcublas","libcudnn")
#         isempty(Libdl.find_library([l], lpath)) && (warn("Cannot find $l");libs=false)
#     end
#     # TODO: eliminate these dependencies:
#     for p in ("CUDArt",) # , "CUBLAS", "CUDNN")
#         isdir(Pkg.dir(p)) || (warn("Cannot find $p");libs=false)
#     end
#     return libs
# end



