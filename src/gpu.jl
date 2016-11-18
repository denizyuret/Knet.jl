macro cuda(lib,fun,x...)
    if Libdl.find_library(["lib$lib"], []) == ""
        msg = "Cannot find lib$lib, please install it and rerun Pkg.build(\"Knet\")."
        :(error($msg))
    else
        fx = Expr(:ccall, ("$fun","lib$lib"), :UInt32, x...)
        msg = "$lib.$fun error "
        err = gensym()
        esc(:(($err=$fx) == 0 || (warn($msg, $err); Base.show_backtrace(STDOUT, backtrace()))))
    end
end

typealias Cptr Ptr{Void}

let GPU=-1, GPUCNT=-1, handles=Dict()
    global gpu, gpuCount, cublashandle, cudnnhandle, cudaRuntimeVersion, cudaDriverVersion

    gpu()=GPU

    function gpuCount() # should not bomb when there is no gpu or nvidia libs
        if GPUCNT == -1
            GPUCNT = try
	        p=Cuint[0]
                # @cuda does not stay quiet so we use ccall here
                # This code is only run once if successful, so nvmlInit here is ok
                eval(:(ccall(("nvmlInit","libnvidia-ml"),UInt32,())==0 || error()))
	        eval(:(ccall(("nvmlDeviceGetCount","libnvidia-ml"),UInt32,(Ptr{Cuint},),$p)==0 || error()))
                # Let us keep nvml initialized for future ops such as meminfo
                # eval(:(ccall(("nvmlShutdown","libnvidia-ml"),UInt32,())==0 || error()))
	        Int(p[1])
            catch
	        0
            end
        end
        return GPUCNT
    end

    function gpu(i::Int)
        (GPU == i) && return i
        if 0 <= i < gpuCount()
            @cuda(cudart,cudaSetDevice, (Cint,), i)
            cublashandle = get!(cublasCreate, handles, (:cublas,i))
            cudnnhandle  = get!(cudnnCreate,  handles, (:cudnn,i))
            cudaRuntimeVersion = (p=Cint[0];@cuda(cudart,cudaRuntimeGetVersion,(Ptr{Cint},),p);Int(p[1]))
            cudaDriverVersion  = (p=Cint[0];@cuda(cudart,cudaDriverGetVersion, (Ptr{Cint},),p);Int(p[1]))
        else
            i = -1
            cublashandle = cudnnhandle = nothing
            # @cuda(cudart,cudaDeviceReset,()) # may still go back and use arrays allocated in a previous gpu
        end
        return (GPU = i)
    end

    function gpu(usegpu::Bool)
        if usegpu && gpuCount() > 0
            pick = free = same = -1
            for i=0:gpuCount()-1
                mem = nvmlDeviceGetMemoryInfo(i)
                if mem[2] > free
                    pick = i
                    free = mem[2]
                    same = 1
                elseif mem[2] == free
                    # pick one of equal devices randomly
                    rand(1:(same+=1)) == 1 && (pick = i)
                end
            end
            gpu(pick)
        else
            for i=0:gpuCount()-1
                @cuda(cudart,cudaDeviceReset,())
            end
            gpu(-1)
        end
    end
end

"""
gpu() returns the id of the active GPU device or -1 if none are active.

gpu(true) resets all GPU devices and activates the one with the most available memory.

gpu(false) resets and deactivates all GPU devices.

gpu(d::Int) activates the GPU device d if 0 <= d < gpuCount(), otherwise deactivates devices.

gpu(true/false) resets all devices.  If there are any allocated
KnetArrays their pointers will be left dangling.  Thus gpu(true/false)
should only be used during startup.  If you want to suspend GPU use
temporarily, use gpu(-1).

gpu(d::Int) does not reset the devices.  You can select a previous
device and find allocated memory preserved.  However trying to operate
on arrays of an inactive device will result in error.

""" gpu

# cudaGetDeviceCount is deprecated, use gpuCount instead:
cudaGetDeviceCount()=(try; p=Cint[0]; eval(:(ccall(("cudaGetDeviceCount","libcudart"),UInt32,(Ptr{Cint},),$p))); p[1]; catch; 0; end) # will not bomb when there is no gpu
cudaGetDevice()=(d=Cint[-1];@cuda(cudart,cudaGetDevice,(Ptr{Cint},),d);d[1])
cudaGetMemInfo()=(f=Csize_t[0];m=Csize_t[0]; @cuda(cudart,cudaMemGetInfo,(Ptr{Csize_t},Ptr{Csize_t}),f,m); (Int(f[1]),Int(m[1])))
cudaDeviceSynchronize()=@cuda(cudart,cudaDeviceSynchronize,())

function nvmlDeviceGetMemoryInfo(i=gpu())
    0 <= i < gpuCount() || return nothing
    dev = Cptr[0]
    mem = Array(Culonglong,3)
    @cuda("nvidia-ml","nvmlDeviceGetHandleByIndex",(Cuint,Ptr{Cptr}),i,dev)
    @cuda("nvidia-ml","nvmlDeviceGetMemoryInfo",(Cptr,Ptr{Culonglong}),dev[1],mem)
    ntuple(i->Int(mem[i]),length(mem))
end

function cublasCreate()
    handleP = Cptr[0]
    @cuda(cublas,cublasCreate_v2, (Ptr{Cptr},), handleP)
    handle = handleP[1]
    atexit(()->@cuda(cublas,cublasDestroy_v2, (Cptr,), handle))
    global cublasVersion = (p=Cint[0];@cuda(cublas,cublasGetVersion_v2,(Cptr,Ptr{Cint}),handle,p);Int(p[1]))
    return handle
end

function cudnnCreate()
    handleP = Cptr[0]
    @cuda(cudnn,cudnnCreate,(Ptr{Cptr},), handleP)
    handle = handleP[1]
    atexit(()->@cuda(cudnn,cudnnDestroy,(Cptr,), handle))
    global cudnnVersion = Int(ccall((:cudnnGetVersion,:libcudnn),Csize_t,()))
    return handle
end

