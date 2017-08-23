let
    # reserve a global state p2pdevs and don't enable p2p twice
    # as enabling p2p access twice raises cuda error
    p2pdevs = Set{Tuple{Int, Int}}()

    global checkP2P, enableP2P

    """
    `checkP2P(did1::Int, did2::Int)::Bool` Checks for peer access between devices did1 and did2.
    Returns false if p2p access is not supported and true otherwise.
    The function checks peer access of both did1 to did2 and did2 to did1; so checkP2P(d1, d2)
    is identical to checkP2P(d2, d1).
    """
    function checkP2P(did1::Int, did2::Int)::Bool
        access = Cint[0]
        @cuda(cudart, cudaDeviceCanAccessPeer, (Ptr{Cint}, Cint, Cint), access, did1, did2)
        res1 = Bool(access[1])
        @cuda(cudart, cudaDeviceCanAccessPeer, (Ptr{Cint}, Cint, Cint), access, did2, did1)
        res2 = Bool(access[1])
        res1 && res2
    end

    """
    `enableP2P(did1::Int, did2::Int)` Enables peer access between devices did1 and did2.
    The function enables peer access of both did1 to did2 and did2 to did1; so enableP2P(d1, d2)
    is identical to enableP2P(d2, d1). When it is called, gpu of the caller thread is preserved.
    """
    function enableP2P(did1::Int, did2::Int)
        if ((did1, did2) in p2pdevs) || ((did2, did1) in p2pdevs)
            return p2pdevs #for consistency
        end
        gpu_temp = gpu()
        gpu(did1)
        @cuda(cudart, cudaDeviceEnablePeerAccess, (Cint, Cint), did2, 0)
        gpu(did2)
        @cuda(cudart, cudaDeviceEnablePeerAccess, (Cint, Cint), did1, 0)
        gpu(gpu_temp)
        push!(p2pdevs, (did1, did2))
    end


    function all_pairs(f::Function, gpuList)
        ngpus = length(gpuList)
        for i = 1:ngpus-1
            for j = (i+1):ngpus
                f(gpuList[i], gpuList[j])
            end
        end
    end

    """
    `enableP2P(gpuList::Union{Array{Int, 1}, Void}=nothing)::Bool`
    Checks and enables peer access between all gpus whose ids are in the gpuList.
    Returns false if peer access does not exist between any gpu pair, true otherwise.
    If gpuList is nothing, then [0:gpuCount()-1] is used as the gpuList.
    """
    function enableP2P(gpuList::Union{Array{Int, 1}, Void}=nothing)::Bool
        if gpuList == nothing
            gpuList = Array{Int, 1}(0:(gpuCount()-1))
        end
        # check the access status
        check = true
        all_pairs(gpuList) do d1, d2
            check = check && checkP2P(d1, d2)
        end
        if ~check; return false; end
        # enable access between peers
        all_pairs(gpuList) do d1, d2
            enableP2P(d1, d2)
        end
        return true
    end

end
