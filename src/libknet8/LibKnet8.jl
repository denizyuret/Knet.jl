# TODO: clean this up. Only @knet8 and @knet8r are needed.

module LibKnet8

export libknet8, @knet8, @knet8r, gpu
using CUDA, Libdl, Pkg.Artifacts
const libknet8 = Libdl.find_library(["libknet8"], @static ((Base.Sys.isapple() && Base.Sys.ARCH==:aarch64) || (occursin("unknown", Base.Sys.MACHINE)) ) ? [] : [artifact"libknet8"])
#DBG const libknet8 = Libdl.find_library(["libknet8"], [@__DIR__])

include("ops.jl")

macro cudacall(lib,fun,returntype,argtypes,argvalues,errmsg=true,notfound=:(error("Cannot find $lib")))
    lib = string(lib); fun = string(fun)
	if isa(argtypes,Expr); argtypes = argtypes.args; end
    if isa(argvalues,Expr); argvalues = argvalues.args; end
	if lib == "cudnn"
		path = CUDA.find_cuda_library(lib,tk,cudnnver)
	elseif lib == "knet8"
		path = libknet8
	else
		path = CUDA.find_cuda_library(lib,tk,tkver)
	end
    if path==nothing || path==""; return notfound; end
    fx = Expr(:call, :ccall, Expr(:tuple,fun,path), returntype, Expr(:tuple,argtypes...), argvalues...)
    r = gensym()
    fx = Expr(:block,esc(:($r=$fx)))
    if errmsg # || TIMER
        # if TIMER
        #     push!(fx.args, esc(:(ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()))))
        # end
        if errmsg
            push!(fx.args, esc(:(if $r!=0; error($(@__MODULE__).getErrorString($lib,$fun,$r)); end)))
        else
            push!(fx.args, esc(r))
        end
        # if TIMER
        #     fx = :(@timeit to $fun ($fx))
        # end
    end
    esc(fx)
end

# macro cudnn(fun, argtypes, argvalues...); :(@cudacall("cudnn",$fun,UInt32,$argtypes,$argvalues)); end
# macro cuda(fun, argtypes, argvalues...); :(@cudacall("cuda",$fun,UInt32,$argtypes,$argvalues)); end
# macro cudart(fun, argtypes, argvalues...); :(@cudacall("cudart",$fun,UInt32,$argtypes,$argvalues)); end
# macro cudart1(fun, argtypes, argvalues...); :(@cudacall("cudart",$fun,UInt32,$argtypes,$argvalues,false,-1)); end # don't throw error
# macro cublas(fun, argtypes, argvalues...); :(@cudacall("cublas",$fun,UInt32,$argtypes,$argvalues)); end
# macro curand(fun, argtypes, argvalues...); :(@cudacall("curand",$fun,UInt32,$argtypes,$argvalues)); end
# macro nvml(fun, argtypes, argvalues...); :(@cudacall("nvml",$fun,UInt32,$argtypes,$argvalues)); end

macro knet8(fun, argtypes, argvalues...)
    fun = "$(fun)_stream"
    push!(argtypes.args, :(CUDA.cudaStream_t))
    argvalues = (argvalues..., :(CUDA.stream()))
    :(@cudacall("knet8",$fun,Nothing,$argtypes,$argvalues,false))
end

macro knet8r(fun, returntype, argtypes, argvalues...)  # specify return type
    fun = "$(fun)_stream"
    push!(argtypes.args, :(CUDA.cudaStream_t))
    argvalues = (argvalues..., :(CUDA.stream()))
    :(@cudacall("knet8",$fun,$returntype,$argtypes,$argvalues,false))
end

function gpu(x...)
    @warn "gpu() is deprecated, please use CUDA.device instead" maxlog=1
    CUDA.functional() ? CUDA.device().handle : -1
end

# TODO: pick best gpu in init, deprecate gpu

# """

# `gpu()` returns the id of the active GPU device or -1 if none are
# active.

# `gpu(true)` resets all GPU devices and activates the one with the most
# available memory.

# `gpu(false)` resets and deactivates all GPU devices.

# `gpu(d::Int)` activates the GPU device `d` if `0 <= d < gpuCount()`,
# otherwise deactivates devices.

# `gpu(true/false)` resets all devices.  If there are any allocated
# KnetArrays their pointers will be left dangling.  Thus
# `gpu(true/false)` should only be used during startup.  If you want to
# suspend GPU use temporarily, use `gpu(-1)`.

# `gpu(d::Int)` does not reset the devices.  You can select a previous
# device and find allocated memory preserved.  However trying to operate
# on arrays of an inactive device will result in error.

# """
# function gpu end

# GPU = -1
# gpu() = GPU
# gpu(d::Integer) = (global GPU; GPU==d ? d : (try CUDA.device!(d); GPU=Int(d); catch; GPU=-1; end))
# gpu(b::Bool) = (global GPU=try b ? Int(CUDA.device().handle) : (CUDA.device_reset!();-1); catch; -1; end) # TODO: pick gpu with most memory

# gpuCount() = length(CUDA.devices())
# gpufree() = Mem.info()[1] + CUDA.pool[].cached_memory()

#=
let GPU=-1, GPUCNT=-1, CUBLAS=nothing, CUDNN=nothing, CURAND=nothing
    global gpu, gpuCount, cublashandle, cudnnhandle, curandGenerator

    gpu()=GPU

    function gpu(usegpu::Bool)
        global cudaRuntimeVersion, cudaDriverVersion, nvmlDriverVersion, nvmlVersion, nvmlfound, cudartfound
        ENV["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # From @ekinakyurek #541
        if !isdefined(@__MODULE__,:cudartfound)
            try #if (cudartfound = (Libdl.find_library(["libcudart"],[]) != ""))
                cudaRuntimeVersion = (p=Cint[0];@cudart(cudaRuntimeGetVersion,(Ptr{Cint},),p);Int(p[1]))
                cudaDriverVersion  = (p=Cint[0];@cudart(cudaDriverGetVersion, (Ptr{Cint},),p);Int(p[1]))
                cudartfound = true
            catch
                cudartfound = false
            end
        end
        if !isdefined(@__MODULE__,:nvmlfound)
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
                        freemem = nvmlDeviceGetMemoryInfo(nvmlid(i))[2]
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
            @cudart(cudaDeviceReset,())  # #541: clear memory
            gpu(pick)
        elseif gpuCount() > 0
            @cudart(cudaDeviceReset,())  # #541: clear memory
            gpu(-1)
        else
            gpu(-1)
        end
    end

    function gpu(i::Int)
        if GPU == i
            # nothing to do
        elseif 0 <= i < gpuCount()
            GPU = i
            @cudart(cudaSetDevice, (Cint,), i)

            # Initializing CUDAnative helps with stability problems for some devices
            # However cuda, nvml and cu use different device numbers! (i) is the cuda device number, CUDAdrv uses cu numbers
            # We find the equivalent cu number from pciBusId:
            # https://stackoverflow.com/questions/13781738/how-does-cuda-assign-device-ids-to-gpus
            # 20200108: CUDAdrv 5.0 initializes to currently active device, so this is no longer needed.
            # CUDAnative.initialize(CUDAdrv.CuDevice(cuid(i)))
            # 20200707: @maleadt added this:
            CUDA.device!(cuid(i))

            # Initialize curand to guard against gpu memory fillup before first dropout (#181)
            curandGenerator()
            
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
                # if nvmlfound
                # # We prefer nvml because cudart takes up memory even if we don't use a device
                #   @nvml(nvmlDeviceGetCount,(Ptr{Cuint},),p)
                # #541: nvml does not respect visible devices (@ekinakyurek)
                # also OSX does not have the nvidia-ml library!
                if cudartfound
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
        dev==-1 && return nothing
        i = dev+2
        if CUBLAS === nothing
            CUBLAS=Array{Any}(nothing,gpuCount()+1)
        end
        if CUBLAS[i] === nothing
            @assert dev == gpu()
            CUBLAS[i]=cublasCreate()
        end
        return CUBLAS[i]
    end

    function cudnnhandle(dev=gpu())
        dev==-1 && return nothing
        i = dev+2
        if CUDNN === nothing
            CUDNN=Array{Any}(nothing,gpuCount()+1)
        end
        if CUDNN[i] === nothing
            @assert dev == gpu()
            CUDNN[i]=cudnnCreate()
        end
        return CUDNN[i]
    end

    function curandGenerator(dev=gpu(); seed=nothing)
        dev==-1 && return nothing
        i = dev+2
        if CURAND === nothing
            CURAND=Array{Any}(nothing,gpuCount()+1)
        end
        if CURAND[i] === nothing || seed !== nothing
            @assert dev == gpu()
            if CURAND[i] !== nothing
                @curand(curandDestroyGenerator,(Cptr,),CURAND[i])
                CURAND[i] = nothing  # to prevent double free if curandCreateGenerator errors
            end
            CURAND[i]=curandCreateGenerator(seed)
        end
        return CURAND[i]
    end
end

# cudaGetDeviceCount is deprecated, use gpuCount instead:
cudaGetDeviceCount()=(p=Cuint[0];@cudart1(cudaGetDeviceCount,(Ptr{Cuint},),p);Int(p[1])) # will not bomb when there is no gpu
cudaGetDevice()=(d=Cint[-1];@cudart(cudaGetDevice,(Ptr{Cint},),d);Int(d[1]))
"Returns free,total memory."
cudaMemGetInfo()=(f=Csize_t[0];m=Csize_t[0]; @cudart(cudaMemGetInfo,(Ptr{Csize_t},Ptr{Csize_t}),f,m); (Int(f[1]),Int(m[1])))
cudaDeviceSynchronize()=@cudart(cudaDeviceSynchronize,())

# cuda, cudart, nvml each number devices differntly
# we are using the cudart ids, here is some converters:

function cudaDeviceGetPCIBusId(i=gpu())
    pciBusId = zeros(UInt8,16)
    @cudart(cudaDeviceGetPCIBusId, (Cptr, Cint, Cint), pciBusId, length(pciBusId), i)
    return unsafe_string(pointer(pciBusId))
end

function cuid(i=gpu())
    pci = cudaDeviceGetPCIBusId(i)
    id = Cint[0]
    @cuda(cuDeviceGetByPCIBusId, (Cptr, Cptr), id, pointer(pci))
    return Int(id[1])
end

function nvmlid(i=gpu())
    pci = cudaDeviceGetPCIBusId(i)
    hnd = Cptr[0]; idx = Cuint[0];
    @nvml(nvmlDeviceGetHandleByPciBusId, (Cptr, Cptr), pointer(pci), hnd)
    @nvml(nvmlDeviceGetIndex, (Cptr, Cptr), hnd[1], idx)
    return Int(idx[1])
end


"Returns total,free,used memory."
function nvmlDeviceGetMemoryInfo(i=nvmlid(gpu()))
    0 <= i || return nothing
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
        return nvmlDeviceGetMemoryInfo(nvmlid(i))
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
    path = CUDA.find_cuda_library("cudnn",tk,cudnnver)
    if path==nothing; error("Cannot find cudnn"); end
    handleP = Cptr[0]
    @cudnn(cudnnCreate,(Ptr{Cptr},), handleP)
    handle = handleP[1]
    atexit(()->@cudnn(cudnnDestroy,(Cptr,), handle))
    global cudnnVersion = Int(eval(:(ccall(("cudnnGetVersion",$path),Csize_t,()))))
    return handle
end

function curandCreateGenerator(seed = nothing)
    CURAND_RNG_PSEUDO_DEFAULT = 100 # Default pseudorandom generator
    r = Cptr[0]
    @curand(curandCreateGenerator,(Cptr,Cint),r,CURAND_RNG_PSEUDO_DEFAULT)
    seed !== nothing && @curand(curandSetPseudoRandomGeneratorSeed,(Cptr,Culonglong),r[1],seed)
    # This call ensures memory buffer allocation. Without it we get "out of memory" later:
    @curand(curandGenerateUniform,(Cptr,Ptr{Cfloat},Csize_t),r[1],C_NULL,0)
    return r[1]
end

const curanderrors = Dict(
    0 => "No errors",
    100 => "Header file and linked library version do not match",
    101 => "Generator not initialized",
    102 => "Memory allocation failed",
    103 => "Generator is wrong type",
    104 => "Argument out of range",
    105 => "Length requested is not a multiple of dimension",
    106 => "GPU does not have double precision required by MRG32k3a",
    201 => "Kernel launch failure",
    202 => "Preexisting failure on library entry",
    203 => "Initialization of CUDA failed",
    204 => "Architecture mismatch, GPU does not support requested feature",
    999 => "Internal library error",
)

const cublaserrors = Dict(
    0 => "CUBLAS_STATUS_SUCCESS",
    1 => "CUBLAS_STATUS_NOT_INITIALIZED",
    3 => "CUBLAS_STATUS_ALLOC_FAILED",
    7 => "CUBLAS_STATUS_INVALID_VALUE",
    8 => "CUBLAS_STATUS_ARCH_MISMATCH",
    11 => "CUBLAS_STATUS_MAPPING_ERROR",
    13 => "CUBLAS_STATUS_EXECUTION_FAILED",
    14 => "CUBLAS_STATUS_INTERNAL_ERROR",
    15 => "CUBLAS_STATUS_NOT_SUPPORTED",
    16 => "CUBLAS_STATUS_LICENSE_ERROR",
)

function getErrorString(lib,fun,ret)
    if lib == "cudnn"
	path = CUDA.find_cuda_library(lib,tk,cudnnver)
    else
	path = CUDA.find_cuda_library(lib,tk,tkver)
    end
    if lib == "cudart" && path != nothing
        str = unsafe_string(@eval(ccall(("cudaGetErrorString",$path),Cstring,(UInt32,),$ret)))
    elseif lib == "cudnn" && path != nothing
        str = unsafe_string(@eval(ccall(("cudnnGetErrorString",$path),Cstring,(UInt32,),$ret)))
    elseif lib == "nvml" && path != nothing
        str = unsafe_string(@eval(ccall(("nvmlErrorString",$path),Cstring,(UInt32,),$ret)))
    elseif lib == "curand"
        str = get(curanderrors,ret,"Unknown $lib error in $fun")
    elseif lib == "cublas"
        str = get(cublaserrors,ret,"Unknown $lib error in $fun")
    else
        str = "Unknown $lib error in $fun"
    end
    string(fun, ": ", ret, ": ", str)
end

=#

#const tk = CUDA.find_toolkit()
#const tkver = isempty(tk) ? v"0" : CUDA.parse_toolkit_version(CUDA.find_cuda_binary("ptxas", tk)) # v"0" because "nothing" will crash precompilation
#const cudnnver = v"7" # cudnn version not read automatically. hardcoded to v7
# const to = TimerOutput()
#function getErrorString end

# TODO: deprecate timer?
# moved profiling option from Knet.jl to gpu.jl to make it self contained for testing
# const TIMER = haskey(ENV,"KNET_TIMER")

end # module
