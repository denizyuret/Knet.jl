export KnetPtr, Cptr, gc
using CUDA: CUDA, CuArray, CuPtr, unsafe_cuMemAlloc_v2, cuMemFree_v2, device, devices, functional, unsafe_free!
const Cptr = Ptr{Cvoid}
const cuallocator = Ref{Bool}(true)
dev() = Int(device().handle)

# KnetPtr type holds a gpu allocated pointer.  We try to minimize the number of actual
# allocations, which are slow, by reusing preallocated but garbage collected pointers.

mutable struct KnetPtr
    ptr                         # actual pointer, removed type ::Cptr for serialization
    len::Int                    # size in bytes
    dev::Int                    # id of the device the pointer belongs to
    parent	                # used to implement shared memory pointers
end

# This is the low level KnetPtr constructor, it adds the finalizer and sets parent to
# nothing, which is only needed for shared pointers.

function KnetPtr(ptr::Cptr,len::Integer,dev::Integer)
    kp = KnetPtr(ptr,len,dev,nothing)
    finalizer(freeKnetPtr, kp)
end

# This constructor is used to create a shared pointer.  We need to keep the parent field to
# prevent premature gc of the parent.  The child does not need a special finalizer.

function KnetPtr(parent::KnetPtr, offs::Integer, len::Integer)
    if len < 0 || offs < 1 || offs+len-1 > parent.len; throw(BoundsError()); end
    KnetPtr(parent.ptr+offs-1, len, parent.dev, parent)
end

# When Julia gc reclaims a KnetPtr object, the following special finalizer does not actually
# release the memory, but inserts it back in the appropriate pool for reuse.

function freeKnetPtr(p::KnetPtr)
    if p.ptr == C_NULL
        # already freed, do nothing
    elseif p.parent isa Nothing
        #@dbg (push!(arraysizes,-p.len); push!(blocksizes,-p.len))
        mem = KnetMems[p.dev+1]
        mem.bfree += p.len
        mem.kfree += 1
        push!(mem.pools[p.len].free, p.ptr)
        p.ptr = C_NULL # to avoid double free by gcnode then gc.
    elseif p.parent isa KnetPtr
        # subarray, do nothing
    else # p.parent isa CuArray
        freeKnetPtrCu(p)
    end
end

# We use the KnetPool type to keep track of allocated and garbage collected pointers: We
# keep one KnetPool struct per size per device.

mutable struct KnetPool
    free::Vector{Cptr}          # pointers available for reuse
    nptr::Int                   # number of allocated pointers
    KnetPool()=new(Cptr[],0)
end

const KNETMEMINIT = 1<<24       # initial gpu memory limit

# KnetMem type keeps memory information for one device.
mutable struct KnetMem
    pools::Dict{Int,KnetPool}   # pointers of a given size
    limit::Int                  # current memory limit
    bytes::Int                  # total bytes allocated (inuse + avail)
    bfree::Int                  # total bytes freed and available
    kptrs::Int                  # number of arrays allocated (inuse + avail)
    kfree::Int                  # number of arrays freed and available
    gc::Int                     # number of times GC.gc called
    knetgc::Int                 # number of times Knet.gc called
    gctime::UInt
end
KnetMem()=KnetMem(Dict{Int,KnetPool}(),KNETMEMINIT,0,0,0,0,0,0,0)

# KnetMems[dev+1] holds memory information for device dev.
KnetMems = nothing
initKnetMems() = (global KnetMems = [ KnetMem() for i in 1:length(devices()) ])
knetmem(dev=dev()) = (if KnetMems == nothing; initKnetMems(); end; KnetMems[dev+1])

# Blocksize determines the actual allocation size given the array size in bytes, and can be
# larger than what the array needs for increased reuse.
# blocksize(n::Int)=n
# function blocksize(n::Int)
#     b = sizeof(n)<<3
#     z = leading_zeros(n-1)
#     1<<(b-z)
# end
blocksize(n::Int,b=cbrt(2))=floor(Int,b^ceil(log(b,n)))

# The following used for debugging and record every request
arraysizes = Int[]; allocs = Int[]; blocksizes = Int[]

# This the main KnetPtr constructor.  It tries to avoid actual allocation which is slow.
# Reusing a preallocated and garbage collected pointer is very fast.
# Allocating a new pointer is about 0.5ms.
# GC.gc() is about 100ms.
# Knet.gc() is about 250ms.

gc_interval() = 2*10^8  # gc interval in ns, optimized on seq2seq model, balancing costs of alloc, GC.gc, Knet.gc
putc(c)=nothing         # putc(c)=print(c) to observe GC.gc, Knet.gc and inclimit

function KnetPtr(arraybytes::Int)
    @assert functional() "Cannot use KnetArray without a GPU."
    cuallocator[] && return KnetPtrCu(arraybytes)
    dev = dev(); @assert dev >= 0 "KnetPtr: bad device id $dev."
    mem = knetmem(dev)
    blockbytes = blocksize(arraybytes)
    #@dbg (push!(arraysizes,arraybytes); push!(blocksizes,blockbytes))
    pool = get!(KnetPool,mem.pools,blockbytes)

    ptr = reuse(mem, pool, blockbytes, 0) # 0. best case we have one available in pool
    ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    if pool.nptr > 0
        ptr = reuse(mem, pool, blockbytes, 1, trygc=false) # 1. try fast gc (~0.5 ms)
        ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    end
    if mem.bytes + blockbytes <= mem.limit # 2. allocate if within limit (~0.5 ms)
        ptr = alloc(mem, pool, blockbytes, 2)
        ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    end
    if pool.nptr > 0 && time_ns() - mem.gctime > gc_interval() # 3. try slow gc (~100 ms) if enough time passed
        ptr = reuse(mem, pool, blockbytes, 3, trygc=true)
        ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    end
    if maybe_inclimit!(mem, max(mem.bytes + blockbytes*2, mem.limit*6÷5)) # 4. try to increase limit
        ptr = alloc(mem, pool, blockbytes, 4)
        ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    end
    if pool.nptr > 0  # 5. try slow gc (~100ms)
        ptr = reuse(mem, pool, blockbytes, 5, trygc=true)
        ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    end
    Knet.gc()  # 6. last ditch effort: ~250 ms + future cost of lost pools
    if mem.bytes + blockbytes <= mem.limit
        ptr = alloc(mem, pool, blockbytes, 6)
        ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    end
    error("Out of gpu memory")
end

function alloc(mem, pool, blockbytes, dbg)
    ptr = knetMalloc(blockbytes)       # ~584μs
    if ptr != nothing
        #@dbg push!(allocs, dbg)
        mem.bytes += blockbytes
        mem.kptrs += 1
        pool.nptr += 1
        return ptr
    else
        return nothing
    end
end

function reuse(mem, pool, blockbytes, dbg; trygc=nothing)
    if trygc !== nothing
        GC.gc(trygc)            # gc(true)=slow, gc(false)=fast
        if trygc
            mem.gctime = time_ns(); mem.gc += 1; putc('-')
        end
    end
    if !isempty(pool.free)
        #@dbg push!(allocs, dbg)
        mem.bfree -= blockbytes
        mem.kfree -= 1
        return pop!(pool.free)
    else
        return nothing
    end
end

function maybe_inclimit!(m::KnetMem, minlimit=m.limit*6÷5)
    maxlimit = m.bytes + gpufree() - 500_000_000 # gpumem()[1] - 500_000_000 # m.bytes + gpufree()
    if m.limit < minlimit <= maxlimit
        putc('^')
        m.limit = minlimit
        @debug kmeminfo()
        return true
    else
        return false
    end
end

KnetPtr(n::Integer)=KnetPtr(Int(n))

# This does the actual allocation, returns `nothing` in case of error
function knetMalloc(nbytes::Int) # 584μs
    ptr = Cptr[0]
    # ret = @cudart1(cudaMalloc,(Ptr{Cptr},Csize_t),ptr,nbytes)
    ret = unsafe_cuMemAlloc_v2(ptr, nbytes)
    ret == 0 ? ptr[1] : nothing
end


"""
    Knet.gc(dev=CUDA.device().handle)

cudaFree all pointers allocated on device `dev` that were previously allocated and garbage
collected. Normally Knet holds on to all garbage collected pointers for reuse. Try this if
you run out of GPU memory.
"""
function gc(dev=dev())
    if KnetMems == nothing; GC.gc(); return; end
    putc('+')
    mem = knetmem(dev)
    mem.knetgc += 1
    GC.gc(); GC.enable(false)
    for (n,v) in mem.pools
        for p in v.free
            # @cudart(cudaFree,(Cptr,),p)
            cuMemFree_v2(CuPtr{Nothing}(UInt(p)))
        end
        v.nptr -= length(v.free)
        mem.kptrs -= length(v.free)
        mem.kfree -= length(v.free)
        mem.bfree -= n * length(v.free)
        mem.bytes -= n * length(v.free)
        empty!(v.free)
    end
    GC.enable(true); GC.gc()
end

function knetgc()
    @warn "knetgc is deprecated, use Knet.gc instead" maxlog=1
    Knet.gc()
end

# Testing the CUDA.jl allocator: set Knet.cuallocator[]=true to use this
function KnetPtrCu(len::Int)
    c = CuArray{UInt8}(undef, len)
    p = convert(Cptr, convert(UInt, Base.unsafe_convert(CuPtr{UInt8}, Base.cconvert(CuPtr{UInt8}, c))))
    kp = KnetPtr(p, len, dev(), c)
    finalizer(freeKnetPtrCu, kp)
end

function freeKnetPtrCu(p::KnetPtr)
    # GC.gc comes here directly, manual calls come through freeKnetPtr()
    # After a manual call, GC.gc may call the finalizer again, avoid double free
    if p.parent isa CuArray
        unsafe_free!(p.parent)
        p.ptr, p.parent = C_NULL, nothing
    elseif p.parent isa Nothing
        @assert p.ptr == C_NULL
        # already freed, do nothing
    elseif p.parent isa KnetPtr
        # subarray, do nothing
    else
        error("Bad parent pointer $(typeof(p.parent))")
    end
end

# Some utilities

using Printf
meminfo(i=dev())=[(k,v.nptr,length(v.free)) for (k,v) in knetmem(i).pools]
kmeminfo(i=dev())=(m=knetmem(i); @sprintf("knetgc=%g gc=%g pools=%g kptrs=%g kfree=%g limit=%g bytes=%g bfree=%g", m.knetgc, m.gc, length(m.pools), m.kptrs, m.kfree, m.limit, m.bytes, m.bfree))

# The following are deprecated, use CUDA.memory_status instead:

# function gpuinfo(msg="",dev=dev();n=10)
#     msg != "" && print("$msg ")
#     if nvmlfound # Libdl.find_library(["libnvidia-ml"],[]) != ""
#         g = nvmlDeviceGetMemoryInfo(nvmlid(dev))
#         println((:dev,dev,:total,g[1],:free,g[2],:used,g[3]))
#     else
#         dev0 = dev(); dev(dev)
#         (free,total) = cudaMemGetInfo()
#         dev(dev0)
#         println((:dev,dev,:total,total,:free,free,:used,total-free))
#     end
#     bytes = bfree = ptrs = k = 0
#     for (s,u,f) in sort(meminfo(dev), by=(x->x[1]*x[2]), rev=true)
#         bytes += s*u; bfree += s*f; ptrs += u; k += 1
#         if k <= n; println((:bytes,s*u,:size,s,:alloc,u,:bfree,f)); end
#     end
#     if n < k; println('⋮'); end
#     println((:totbytes,bytes,:bfree,bfree,:ptrs,ptrs))
#     if KnetMems != nothing
#         mem = KnetMems[dev+1]
#         println((:membytes,mem.bytes,:bfree,mem.bfree,:limit,mem.limit,:pools,length(mem.pools)))
#     end
# end

# function memdbg(msg="")
#     if nvmlfound # Libdl.find_library(["libnvidia-ml"],[]) != ""
#         m = nvmlDeviceGetMemoryInfo()
#     else
#         m = (0,0,0)
#     end
#     c = cudaMemGetInfo()
#     x = [0,0]
#     for (s,u,f) in meminfo()
#         x[1] += s*u
#         x[2] += s*f
#     end
#     println("""$msg:
# cudaMemGetInfo: ctotal: $(c[2]) cfree: $(c[1]) ctotal-cfree: $(c[2]-c[1])
# nvmlDeviceGetMemoryInfo: ntotal: $(m[1]) nfree: $(m[2]) nused: $(m[3]) nfree+used: $(m[2]+m[3])
# KnetPtr: ktotal: $(x[1]) kavail: $(x[2]) ktotal-kavail: $(x[1]-x[2])
# nused-ktotal: $(m[3]-x[1])
# """)
# end

