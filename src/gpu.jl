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

typealias Cptr Ptr{Void}

let GPU=-1, handles=Dict()
    global gpu, cublashandle, cudnnhandle, cudaRuntimeVersion, cudaDriverVersion

    gpu()=GPU

    function gpu(i::Int)
        (GPU == i) && return i
        if 0 <= i < cudaGetDeviceCount()
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

    function gpu(b::Bool)
        if b
            pick = mem = same = -1
            f = Csize_t[0]; m = Csize_t[0]
            for i=0:cudaGetDeviceCount()-1
                @cuda(cudart,cudaSetDevice,(Cint,),i)
                @cuda(cudart,cudaMemGetInfo,(Ptr{Csize_t},Ptr{Csize_t}),f,m)
                @cuda(cudart,cudaDeviceReset,())
                if f[1] > mem
                    pick = i
                    mem = f[1]
                    same = 1
                elseif f[1] == mem
                    # pick one of equal devices randomly
                    rand(1:(same+=1)) == 1 && (pick = i)
                end
            end
            gpu(pick)
        else
            for i=0:cudaGetDeviceCount()-1
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

gpu(d::Int) activates the GPU device d if 0 <= d < cudaGetDeviceCount(), otherwise deactivates devices.

gpu(true/false) resets all devices.  If there are any allocated
KnetArrays their pointers will be left dangling.  Thus gpu(true/false)
should only be used during startup.  If you want to suspend GPU use
temporarily, use gpu(-1).

gpu(d::Int) does not reset the devices.  You can select a previous
device and find allocated memory preserved.  However trying to operate
on arrays of an inactive device will result in error.

""" gpu

cudaGetDeviceCount()=(try; p=Cint[0]; eval(:(ccall(("cudaGetDeviceCount","libcudart"),UInt32,(Ptr{Cint},),$p))); p[1]; catch; 0; end) # will not bomb when there is no gpu
cudaGetDevice()=(d=Cint[-1];@cuda(cudart,cudaGetDevice,(Ptr{Cint},),d);d[1])
cudaGetMemInfo()=(f=Csize_t[0];m=Csize_t[0]; @cuda(cudart,cudaMemGetInfo,(Ptr{Csize_t},Ptr{Csize_t}),f,m); (Int(f[1]),Int(m[1])))
cudaDeviceSynchronize()=@cuda(cudart,cudaDeviceSynchronize,())

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

