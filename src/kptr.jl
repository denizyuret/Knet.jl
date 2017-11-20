GCDEBUG=false; gcdebug(b::Bool)=(global GCDEBUG=b)

# KnetPtr type holds a gpu (dev>=0) or cpu (dev=-1) allocated pointer.
# We try to minimize the number of actual allocations, which are slow,
# by reusing preallocated but garbage collected pointers.

type KnetPtr
    ptr::Cptr                   # actual pointer
    len::Int                    # size in bytes
    dev::Int                    # id of the device the pointer belongs to
    parent                      # used to implement shared memory pointers
end

# We use the KnetPtrs type to keep track of allocated and garbage
# collected pointers: We keep one KnetPtrs struct per size per device.

type KnetPtrs
    used::Int                   # number of allocated pointers
    free::Array{Cptr,1}         # pointers available for reuse
    KnetPtrs()=new(0,Array{Cptr}(0))
end

# KnetFree[dev+2] will hold a dictionary from sizes to KnetPtrs for
# device dev.  KnetFree[1] reserved for the cpu in case we decide to
# support it in the future.  It is initialized in KnetPtr(nbytes).

KnetFree = nothing
initKnetFree()=(global KnetFree=[ Dict{Int,KnetPtrs}() for i=1:gpuCount()+1 ])

# When Julia gc reclaims a KnetPtr object, the following special
# finalizer does not actually release the memory, but inserts it in
# the KnetFree[dev+2] dict keyed by length in bytes so it can be
# reused.

function freeKnetPtr(p::KnetPtr)
    ptrs = KnetFree[p.dev+2][p.len]
    push!(ptrs.free,p.ptr)
end

# This is the low level KnetPtr constructor, it adds the finalizer and
# sets parent to `nothing` which is only needed for shared pointers.

function KnetPtr(ptr::Cptr,len::Int,dev::Int)
    kp = KnetPtr(ptr,len,dev,nothing)
    finalizer(kp, freeKnetPtr)
    return kp
end

# This constructor is used to create a shared pointer.  We need to
# keep the parent field to prevent premature gc of the parent.  The
# child does not need a special finalizer.

function KnetPtr(parent::KnetPtr, offs::Integer, len::Integer)
    if len < 0 || offs < 1 || offs+len-1 > parent.len
        throw(BoundsError())
    end
    KnetPtr(parent.ptr+offs-1, len, parent.dev, parent)
end

# This the main KnetPtr constructor.  It tries to avoid actual
# allocation which is slow.  First it tries to find a previously
# allocated and garbage collected pointer in KnetFree[dev+2].  If not
# available it tries to allocate a new one (about 10 μs).  Otherwise
# it tries running gc() and see if we get a pointer back (about 75
# ms).  Finally if all else fails, it calls knetgc which cleans up all
# allocated and garbage collected KnetPtrs on the current device and
# tries allocation one last time.

function KnetPtr(nbytes::Integer)
    KnetFree==nothing && initKnetFree()
    dev = gpu()
    if dev < 0
        error("KnetPtr: bad device id $dev.")
    end
    ptrs = get!(KnetPtrs,KnetFree[dev+2],nbytes)
    if !isempty(ptrs.free)
        return KnetPtr(pop!(ptrs.free),nbytes,dev)
    end
    ptr = knetMalloc(nbytes)
    if ptr != nothing
        ptrs.used += 1
        return KnetPtr(ptr,nbytes,dev)
    end
    gc(); if GCDEBUG; print('-'); end
    if !isempty(ptrs.free)
        return KnetPtr(pop!(ptrs.free),nbytes,dev)
    end
    knetgc(); if GCDEBUG; print('+'); end
    ptr = knetMalloc(nbytes)
    if ptr != nothing
        ptrs.used += 1
        return KnetPtr(ptr,nbytes,dev)
    end
    error("Out of gpu memory")
end

# This does the actual allocation, returns `nothing` in case of error
function knetMalloc(nbytes::Int)
    # we no longer support cpu pointers, all overloaded ops rely on KnetPtr being on a GPU
    # gpu() >= 0 || return(convert(Cptr, pointer(Array{UInt8}(nbytes))))
    ptr = Cptr[0]
    ret = @cuda1(cudart,cudaMalloc,(Ptr{Cptr},Csize_t),ptr,nbytes)
    if ret == 0
        return ptr[1]
    else # error("cudaMalloc($nbytes) error $ret")
        return nothing
    end
end


"""

    knetgc(dev=gpu())

cudaFree all pointers allocated on device `dev` that were previously
allocated and garbage collected. Normally Knet holds on to all garbage
collected pointers for reuse. Try this if you run out of GPU memory.

"""
function knetgc(dev=gpu())
    if KnetFree == nothing; return; end
    gc(); gc_enable(false)
    for v in values(KnetFree[dev+2])
        if dev >= 0
            for p in v.free
                @cuda(cudart,cudaFree,(Cptr,),p)
            end
        end
        v.used -= length(v.free)
        empty!(v.free)
    end
    gc_enable(true); gc()
end

# Some utilities
meminfo(i=gpu())=(KnetFree==nothing ? [] : [(k,v.used,length(v.free)) for (k,v) in KnetFree[i+2]])
gpufree(i=gpu())=nvmlDeviceGetMemoryInfo(i)[2]

function gpuinfo(msg="",dev=gpu();n=10)
    msg != "" && print("$msg ")
    g = nvmlDeviceGetMemoryInfo(dev)
    println((:dev,dev,:total,g[1],:free,g[2],:used,g[3]))
    total = k = 0
    for (s,u,f) in sort(meminfo(dev), by=(x->x[1]*x[2]), rev=true)
        total += s*u; k += 1
        if k <= n; println((:total,s*u,:size,s,:alloc,u,:avail,f)); end
    end
    if n < k; println('⋮'); end
    println((:final,total))
end

function memdbg(msg="")
    m = nvmlDeviceGetMemoryInfo()
    c = cudaGetMemInfo()
    x = [0,0]
    for (s,u,f) in meminfo()
        x[1] += s*u
        x[2] += s*f
    end
    println("""$msg:
cudaGetMemInfo: ctotal: $(c[2]) cfree: $(c[1]) ctotal-cfree: $(c[2]-c[1])
nvmlDeviceGetMemoryInfo: ntotal: $(m[1]) nfree: $(m[2]) nused: $(m[3]) nfree+used: $(m[2]+m[3])
KnetPtr: ktotal: $(x[1]) kavail: $(x[2]) ktotal-kavail: $(x[1]-x[2])
nused-ktotal: $(m[3]-x[1])
""")
end
