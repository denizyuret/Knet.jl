macro gpu(_ex); if gpu()>=0; esc(_ex); end; end

macro cuda(lib,fun,x...)        # give an error if library missing, or if error code!=0
    if Libdl.find_library(["lib$lib"], []) != ""
        if VERSION >= v"0.6.0"
            fx = Expr(:call, :ccall, ("$fun","lib$lib"), :UInt32, x...)
        else
            fx = Expr(:ccall, ("$fun","lib$lib"), :UInt32, x...)
        end
        msg = "$lib.$fun error "
        err = gensym()
        # esc(:(if ($err=$fx) != 0; warn($msg, $err); Base.show_backtrace(STDOUT, backtrace()); end))
        esc(:(if ($err=$fx) != 0; error($msg, $err); end; Knet.@gs))
    else
        Expr(:call,:error,"Cannot find lib$lib, please install it and rerun Pkg.build(\"Knet\").")
    end
end

macro cuda1(lib,fun,x...)       # return -1 if library missing, error code if run
    if Libdl.find_library(["lib$lib"], []) != ""
        if VERSION >= v"0.6.0"
            fx = Expr(:call, :ccall, ("$fun","lib$lib"), :UInt32, x...)
        else
            fx = Expr(:ccall, ("$fun","lib$lib"), :UInt32, x...)
        end
        err = gensym()
        esc(:($err=$fx; Knet.@gs; $err))
    else
        -1
    end
end

macro knet8(fun,x...)       # error if libknet8 missing, nothing if run
    if libknet8 != ""
        if VERSION >= v"0.6.0"
            fx = Expr(:call, :ccall, ("$fun",libknet8), :Void, x...)
        else
            fx = Expr(:ccall, ("$fun",libknet8), :Void, x...)
        end
        err = gensym()
        esc(:($err=$fx; Knet.@gs; $err))
    else
        Expr(:call,:error,"Cannot find libknet8, please rerun Pkg.build(\"Knet\").")
    end
end

const Cptr = Ptr{Void}

"""

`gpu()` returns the id of the active GPU device or -1 if none are
active.

`gpu(true)` resets all GPU devices and activates the one with the most
available memory.

`gpu(false)` resets and deactivates all GPU devices.

`gpu(d::Int)` activates the GPU device `d` if `0 <= d < gpuCount()`,
otherwise deactivates devices.

`gpu(true/false)` resets all devices.  If there are any allocated
KnetArrays their pointers will be left dangling.  Thus
`gpu(true/false)` should only be used during startup.  If you want to
suspend GPU use temporarily, use `gpu(-1)`.

`gpu(d::Int)` does not reset the devices.  You can select a previous
device and find allocated memory preserved.  However trying to operate
on arrays of an inactive device will result in error.

"""
function gpu end

let GPU=-1, GPUCNT=-1, CUBLAS=nothing, CUDNN=nothing
    global gpu, gpuCount, cublashandle, cudnnhandle

    gpu()=GPU

    function gpu(usegpu::Bool)
        global cudaRuntimeVersion, cudaDriverVersion, nvmlDriverVersion, nvmlVersion, nvmlfound, cudartfound
        if !isdefined(:cudartfound)
            try #if (cudartfound = (Libdl.find_library(["libcudart"],[]) != ""))
                cudaRuntimeVersion = (p=Cint[0];@cuda(cudart,cudaRuntimeGetVersion,(Ptr{Cint},),p);Int(p[1]))
                cudaDriverVersion  = (p=Cint[0];@cuda(cudart,cudaDriverGetVersion, (Ptr{Cint},),p);Int(p[1]))
                cudartfound = true
            catch
                cudartfound = false
            end
        end
        if !isdefined(:nvmlfound)
            try #if (nvmlfound = (Libdl.find_library(["libnvidia-ml"],[]) != ""))
                # This code is only run once if successful, so nvmlInit here is ok
                @cuda("nvidia-ml",nvmlInit,())
                s = zeros(UInt8,80)
                @cuda("nvidia-ml",nvmlSystemGetDriverVersion,(Ptr{Cchar},Cuint),s,80)
                nvmlDriverVersion = unsafe_string(pointer(s))
                @cuda("nvidia-ml",nvmlSystemGetNVMLVersion,(Ptr{Cchar},Cuint),s,80)
                nvmlVersion = unsafe_string(pointer(s))
                # Let us keep nvml initialized for future ops such as meminfo
                # @cuda("nvidia-ml",nvmlShutdown,())
                nvmlfound = true
            catch
                nvmlfound = false
            end
        end
        if usegpu && gpuCount() > 0
            if GPU >= 0
                return
            elseif gpuCount() == 1
                pick = 0
            else # Pick based on memory usage
                pick = free = same = -1
                for i=0:gpuCount()-1
                    if nvmlfound
                        freemem = nvmlDeviceGetMemoryInfo(i)[2]
                    else
                        @cuda(cudart,cudaSetDevice, (Cint,), i)
                        freemem = cudaMemGetInfo()[1]
                        @cuda(cudart,cudaDeviceReset,()) # otherwise we leave a memory footprint
                    end
                    if freemem > free
                        pick = i
                        free = freemem
                        same = 1
                    elseif freemem == free
                        # pick one of equal devices randomly
                        rand(1:(same+=1)) == 1 && (pick = i)
                    end
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

    function gpu(i::Int)
        if GPU == i
            # nothing to do 
        elseif 0 <= i < gpuCount()
            GPU = i
            @cuda(cudart,cudaSetDevice, (Cint,), i)
            # Initialize curand to guard against gpu memory fillup before first dropout (#181)
            curandInit()
        else
            GPU = -1
            # @cuda(cudart,cudaDeviceReset,()) # may still go back and use arrays allocated in a previous gpu
        end
        return GPU
    end

    function gpuCount() # should not bomb when there is no gpu or nvidia libs
        if GPUCNT == -1
            GPUCNT = try
	        p=Cuint[0]
                if nvmlfound
                    # @cuda does not stay quiet so we use @cuda1 here
                    @cuda1("nvidia-ml",nvmlDeviceGetCount,(Ptr{Cuint},),p)
                elseif cudartfound
                    # OSX does not have the nvidia-ml library!
                    # We prefer nvml because cudart takes up memory even if we don't use a device
                    @cuda1(cudart,cudaGetDeviceCount,(Ptr{Cuint},),p)
                end
	        Int(p[1])
            catch
	        0
            end
        end
        return GPUCNT
    end

    function cublashandle(dev=gpu())
        if dev==-1; error("No cublashandle for CPU"); end
        i = dev+2
        if CUBLAS == nothing; CUBLAS=Array{Any}(gpuCount()+1); end
        if !isassigned(CUBLAS,i); CUBLAS[i]=cublasCreate(); end
        return CUBLAS[i]
    end

    function cudnnhandle(dev=gpu())
        if dev==-1; error("No cudnnhandle for CPU"); end
        i = dev+2
        if CUDNN == nothing; CUDNN=Array{Any}(gpuCount()+1); end
        if !isassigned(CUDNN,i); CUDNN[i]=cudnnCreate(); end
        return CUDNN[i]
    end
end

# cudaGetDeviceCount is deprecated, use gpuCount instead:
cudaGetDeviceCount()=(try; p=Cint[0]; eval(:(ccall(("cudaGetDeviceCount","libcudart"),UInt32,(Ptr{Cint},),$p))); p[1]; catch; 0; end) # will not bomb when there is no gpu
cudaGetDevice()=(d=Cint[-1];@cuda(cudart,cudaGetDevice,(Ptr{Cint},),d);d[1])
"Returns free,total memory."
cudaMemGetInfo()=(f=Csize_t[0];m=Csize_t[0]; @cuda(cudart,cudaMemGetInfo,(Ptr{Csize_t},Ptr{Csize_t}),f,m); (Int(f[1]),Int(m[1])))
cudaDeviceSynchronize()=@cuda(cudart,cudaDeviceSynchronize,())

"Returns total,free,used memory."
function nvmlDeviceGetMemoryInfo(i=gpu())
    0 <= i < gpuCount() || return nothing
    dev = Cptr[0]
    mem = Array{Culonglong}(3)
    @cuda("nvidia-ml","nvmlDeviceGetHandleByIndex",(Cuint,Ptr{Cptr}),i,dev)
    @cuda("nvidia-ml","nvmlDeviceGetMemoryInfo",(Cptr,Ptr{Culonglong}),dev[1],mem)
    ntuple(i->Int(mem[i]),length(mem))
end

function gpumem(i=gpu())
    if i < 0 || i >= gpuCount()
        return (0,0,0)
    elseif nvmlfound
        return nvmlDeviceGetMemoryInfo(i)
    elseif cudartfound
        dev=gpu(); gpu(i)
        free,total = cudaMemGetInfo()
        gpu(dev)
        return (total,free,total-free)
    else
        error("gpumem cannot find nvml or cudart.")
    end
end

gpufree(i=gpu())=gpumem(i)[2]

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

function curandInit()
    p = Cptr[0]; r = Cptr[0]; 
    @cuda(cudart,cudaMalloc,(Ptr{Cptr},Csize_t),p,sizeof(Float32))
    @cuda(curand,curandCreateGenerator,(Cptr,Cint),r,100)
    @cuda(curand,curandGenerateUniform,(Cptr,Ptr{Cfloat},Csize_t),r[1],p[1],1)
    @cuda(curand,curandDestroyGenerator,(Cptr,),r[1])
    @cuda(cudart,cudaFree,(Cptr,),p[1])
end
