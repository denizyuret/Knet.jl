using CUDAapi, TimerOutputs
const tk = find_toolkit()
const to = TimerOutput()
const Cptr = Ptr{Cvoid}

# moved profiling option from Knet.jl to gpu.jl to make it self contained for testing
const TIMER = haskey(ENV,"KNET_TIMER")

macro cudacall(lib,fun,returntype,argtypes,argvalues,errstr="",notfound=:(error("Cannot find $lib")))
    lib = string(lib); fun = string(fun); errstr=string(errstr)
    if isa(argtypes,Expr); argtypes = argtypes.args; end
    if isa(argvalues,Expr); argvalues = argvalues.args; end
    path = (lib=="knet8" ? libknet8 : find_cuda_library(lib,tk))
    if path==nothing || path==""; return notfound; end
    fx = Expr(:call, :ccall, Expr(:tuple,fun,path), returntype, Expr(:tuple,argtypes...), argvalues...)
    r = gensym()
    fx = Expr(:block,esc(:($r=$fx)))
    if errstr != "" || TIMER
        if TIMER
            push!(fx.args, esc(:(ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()))))
        end
        if errstr != ""
            push!(fx.args, esc(:(if $r!=0; error(unsafe_string(ccall(($errstr,$path),Cstring,(UInt8,),$r))); end)))
        else
            push!(fx.args, esc(r))
        end
        if TIMER
            fx = :(@timeit to $fun ($fx))
        end
    end
    esc(fx)
end

macro cudnn(fun, argtypes, argvalues...); :(@cudacall("cudnn",$fun,UInt32,$argtypes,$argvalues,"cudnnGetErrorString")); end
macro cudart(fun, argtypes, argvalues...); :(@cudacall("cudart",$fun,UInt32,$argtypes,$argvalues,"cudaGetErrorString")); end
macro cudart1(fun, argtypes, argvalues...); :(@cudacall("cudart",$fun,UInt32,$argtypes,$argvalues,"",-1)); end # don't throw error
macro cublas(fun, argtypes, argvalues...); :(@cudacall("cublas",$fun,UInt32,$argtypes,$argvalues,"cudaGetErrorString")); end
macro curand(fun, argtypes, argvalues...); :(@cudacall("curand",$fun,UInt32,$argtypes,$argvalues,"cudaGetErrorString")); end
macro nvml(fun, argtypes, argvalues...); :(@cudacall("nvml",$fun,UInt32,$argtypes,$argvalues,"nvmlErrorString")); end
macro knet8(fun, argtypes, argvalues...); :(@cudacall("knet8",$fun,Nothing,$argtypes,$argvalues)); end
macro knet8r(fun, returntype, argtypes, argvalues...); :(@cudacall("knet8",$fun,$returntype,$argtypes,$argvalues)); end # specify return type


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
        if !isdefined(Knet,:cudartfound)
            try #if (cudartfound = (Libdl.find_library(["libcudart"],[]) != ""))
                cudaRuntimeVersion = (p=Cint[0];@cudart(cudaRuntimeGetVersion,(Ptr{Cint},),p);Int(p[1]))
                cudaDriverVersion  = (p=Cint[0];@cudart(cudaDriverGetVersion, (Ptr{Cint},),p);Int(p[1]))
                cudartfound = true
            catch
                cudartfound = false
            end
        end
        if !isdefined(Knet,:nvmlfound)
            try #if (nvmlfound = (Libdl.find_library(["libnvidia-ml"],[]) != ""))
                # This code is only run once if successful, so nvmlInit here is ok
                @nvml(nvmlInit,())
                s = zeros(UInt8,80)
                @nvml(nvmlSystemGetDriverVersion,(Ptr{Cchar},Cuint),s,80)
                nvmlDriverVersion = unsafe_string(pointer(s))
                @nvml(nvmlSystemGetNVMLVersion,(Ptr{Cchar},Cuint),s,80)
                nvmlVersion = unsafe_string(pointer(s))
                # Let us keep nvml initialized for future ops such as meminfo
                # @nvml(nvmlShutdown,())
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
                        @cudart(cudaSetDevice, (Cint,), i)
                        freemem = cudaMemGetInfo()[1]
                        @cudart(cudaDeviceReset,()) # otherwise we leave a memory footprint
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
                @cudart(cudaDeviceReset,())
            end
            gpu(-1)
        end
    end

    function gpu(i::Int)
        if GPU == i
            # nothing to do 
        elseif 0 <= i < gpuCount()
            GPU = i
            @cudart(cudaSetDevice, (Cint,), i)
            # Initialize curand to guard against gpu memory fillup before first dropout (#181)
            curandInit()
        else
            GPU = -1
            # @cudart(cudaDeviceReset,()) # may still go back and use arrays allocated in a previous gpu
        end
        return GPU
    end

    function gpuCount() # should not bomb when there is no gpu or nvidia libs
        if GPUCNT == -1
            GPUCNT = try
	        p=Cuint[0]
                if nvmlfound
                    @nvml(nvmlDeviceGetCount,(Ptr{Cuint},),p)
                elseif cudartfound
                    # OSX does not have the nvidia-ml library!
                    # We prefer nvml because cudart takes up memory even if we don't use a device
                    @cudart1(cudaGetDeviceCount,(Ptr{Cuint},),p)
                end
	        Int(p[1])
            catch
	        0
            end
        end
        return GPUCNT
    end

    function cublashandle(dev=gpu())
        if dev==-1
            # error("No cublashandle for CPU")
            return nothing
        end
        i = dev+2
        if CUBLAS == nothing; CUBLAS=Array{Any}(undef,gpuCount()+1); end
        if !isassigned(CUBLAS,i); CUBLAS[i]=cublasCreate(); end
        return CUBLAS[i]
    end

    function cudnnhandle(dev=gpu())
        if dev==-1
            # error("No cudnnhandle for CPU")
            return nothing
        end
        i = dev+2
        if CUDNN == nothing; CUDNN=Array{Any}(undef,gpuCount()+1); end
        if !isassigned(CUDNN,i); CUDNN[i]=cudnnCreate(); end
        return CUDNN[i]
    end
end

# cudaGetDeviceCount is deprecated, use gpuCount instead:
cudaGetDeviceCount()=(p=Cuint[0];@cudart1(cudaGetDeviceCount,(Ptr{Cuint},),p);Int(p[1])) # will not bomb when there is no gpu
cudaGetDevice()=(d=Cint[-1];@cudart(cudaGetDevice,(Ptr{Cint},),d);Int(d[1]))
"Returns free,total memory."
cudaMemGetInfo()=(f=Csize_t[0];m=Csize_t[0]; @cudart(cudaMemGetInfo,(Ptr{Csize_t},Ptr{Csize_t}),f,m); (Int(f[1]),Int(m[1])))
cudaDeviceSynchronize()=@cudart(cudaDeviceSynchronize,())

"Returns total,free,used memory."
function nvmlDeviceGetMemoryInfo(i=gpu())
    0 <= i < gpuCount() || return nothing
    dev = Cptr[0]
    mem = Array{Culonglong}(undef,3)
    @nvml("nvmlDeviceGetHandleByIndex",(Cuint,Ptr{Cptr}),i,dev)
    @nvml("nvmlDeviceGetMemoryInfo",(Cptr,Ptr{Culonglong}),dev[1],mem)
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
    @cublas(cublasCreate_v2, (Ptr{Cptr},), handleP)
    handle = handleP[1]
    atexit(()->@cublas(cublasDestroy_v2, (Cptr,), handle))
    global cublasVersion = (p=Cint[0];@cublas(cublasGetVersion_v2,(Cptr,Ptr{Cint}),handle,p);Int(p[1]))
    return handle
end

function cudnnCreate()
    path = find_cuda_library("cudnn",tk)
    if path==nothing; error("Cannot find cudnn"); end
    handleP = Cptr[0]
    @cudnn(cudnnCreate,(Ptr{Cptr},), handleP)
    handle = handleP[1]
    atexit(()->@cudnn(cudnnDestroy,(Cptr,), handle))
    global cudnnVersion = Int(eval(:(ccall(("cudnnGetVersion",$path),Csize_t,()))))
    return handle
end

function curandInit()
    p = Cptr[0]; r = Cptr[0]; 
    @cudart(cudaMalloc,(Ptr{Cptr},Csize_t),p,sizeof(Float32))
    @curand(curandCreateGenerator,(Cptr,Cint),r,100)
    @curand(curandGenerateUniform,(Cptr,Ptr{Cfloat},Csize_t),r[1],p[1],1)
    @curand(curandDestroyGenerator,(Cptr,),r[1])
    @cudart(cudaFree,(Cptr,),p[1])
end
