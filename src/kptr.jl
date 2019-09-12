# KnetPtr type holds a gpu allocated pointer.  We try to minimize the number of actual
# allocations, which are slow, by reusing preallocated but garbage collected pointers.

mutable struct KnetPtr
    ptr                         # actual pointer, removed type ::Cptr for serialization
    len::Int                    # size in bytes
    dev::Int                    # id of the device the pointer belongs to
    parent::KnetPtr             # used to implement shared memory pointers

    # This is the low level KnetPtr constructor, it adds the finalizer and
    # does not assign parent which is only needed for shared pointers.

    function KnetPtr(ptr::Cptr,len::Int,dev::Int)
        kp = new(ptr,len,dev)
        finalizer(freeKnetPtr, kp)
    end

    # This constructor is used to create a shared pointer.  We need to
    # keep the parent field to prevent premature gc of the parent.  The
    # child does not need a special finalizer.

    function KnetPtr(parent::KnetPtr, offs::Int, len::Int)
        if len < 0 || offs < 1 || offs+len-1 > parent.len; throw(BoundsError()); end
        new(parent.ptr+offs-1, len, parent.dev, parent)
    end

    # This one is used by serialize:
    KnetPtr(ptr::Array{UInt8},len::Int)=new(ptr,len,-1)

end

# When Julia gc reclaims a KnetPtr object, the following special finalizer does not actually
# release the memory, but inserts it back in the appropriate pool for reuse.

function freeKnetPtr(p::KnetPtr)
    if p.ptr == C_NULL || isdefined(p,:parent); return; end
    mem = KnetMems[p.dev+1]
    mem.bfree += p.len
    mem.kfree += 1
    push!(mem.pools[p.len].free, p.ptr)
    p.ptr = C_NULL # to avoid double free by gcnode then gc.
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
initKnetMems() = (global KnetMems = [ KnetMem() for i in 1:gpuCount() ])
knetmem(dev=gpu()) = (if KnetMems == nothing; initKnetMems(); end; KnetMems[dev+1])

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
    dev = gpu(); @assert dev >= 0 "KnetPtr: bad device id $dev."
    mem = knetmem(dev)
    blockbytes = blocksize(arraybytes)
    @dbg (push!(arraysizes,arraybytes); push!(blocksizes,blockbytes))
    pool = get!(KnetPool,mem.pools,blockbytes)

    ptr = reuse(mem, pool, blockbytes, 1) # 1. best case we have one available in pool
    ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    if pool.nptr > 0 && time_ns() - mem.gctime > gc_interval() # 2. try gc (~100 ms) if we had allocated this size before and enough time passed
        ptr = reuse(mem, pool, blockbytes, 2, trygc=true); putc('-')
        ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    end
    if mem.bytes + blockbytes <= mem.limit # 3. allocate if within limit
        ptr = alloc(mem, pool, blockbytes, 3)
        ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    end
    if maybe_inclimit!(mem, max(mem.bytes + blockbytes*2, mem.limit*6÷5)) # 4. try to increase limit
        ptr = alloc(mem, pool, blockbytes, 4); putc('^')
        ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    end
    ### This does not prevent Knet.gc for too long:
    # if pool.nptr > 0 && time_ns() - mem.gctime > gc_interval2()  # 5. One last gc before Knet.gc bomb
    #     ptr = reuse(mem, pool, blockbytes, 5, trygc=true); putc('=')
    #     ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    # end
    Knet.gc(); putc('+')  # 6. last ditch effort: ~250 ms + future cost of lost pools
    if mem.bytes + blockbytes <= mem.limit
        ptr = alloc(mem, pool, blockbytes, 6)
        ptr != nothing && return KnetPtr(ptr,blockbytes,dev)
    end
    error("Out of gpu memory")
end

function alloc(mem, pool, blockbytes, dbg)
    ptr = knetMalloc(blockbytes)       # ~584μs
    if ptr != nothing
        @dbg push!(allocs, dbg)
        mem.bytes += blockbytes
        mem.kptrs += 1
        pool.nptr += 1
        return ptr
    else
        return nothing
    end
end

function reuse(mem, pool, blockbytes, dbg; trygc=false)
    if trygc
        if TIMER; @timeit to "gc" GC.gc(); end
        mem.gctime = time_ns(); mem.gc += 1
    end
    if !isempty(pool.free)
        @dbg push!(allocs, dbg)
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
    ret = @cudart1(cudaMalloc,(Ptr{Cptr},Csize_t),ptr,nbytes)
    ret == 0 ? ptr[1] : nothing
end


"""
    Knet.gc(dev=gpu())

cudaFree all pointers allocated on device `dev` that were previously allocated and garbage
collected. Normally Knet holds on to all garbage collected pointers for reuse. Try this if
you run out of GPU memory.
"""
function gc(dev=gpu())
    if KnetMems == nothing; return; end
    mem = knetmem(dev)
    mem.knetgc += 1
    GC.gc(); GC.enable(false)
    for (n,v) in mem.pools
        for p in v.free
            @cudart(cudaFree,(Cptr,),p)
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

# Some utilities

meminfo(i=gpu())=[(k,v.nptr,length(v.free)) for (k,v) in knetmem(i).pools]

using Printf
kmeminfo(i=gpu())=(m=knetmem(i); @sprintf("knetgc=%g gc=%g pools=%g kptrs=%g kfree=%g limit=%g bytes=%g bfree=%g", m.knetgc, m.gc, length(m.pools), m.kptrs, m.kfree, m.limit, m.bytes, m.bfree))

function gpuinfo(msg="",dev=gpu();n=10)
    msg != "" && print("$msg ")
    if nvmlfound # Libdl.find_library(["libnvidia-ml"],[]) != ""
        g = nvmlDeviceGetMemoryInfo(nvmlid(dev))
        println((:dev,dev,:total,g[1],:free,g[2],:used,g[3]))
    else
        dev0 = gpu(); gpu(dev)
        (free,total) = cudaMemGetInfo()
        gpu(dev0)
        println((:dev,dev,:total,total,:free,free,:used,total-free))
    end
    bytes = bfree = ptrs = k = 0
    for (s,u,f) in sort(meminfo(dev), by=(x->x[1]*x[2]), rev=true)
        bytes += s*u; bfree += s*f; ptrs += u; k += 1
        if k <= n; println((:bytes,s*u,:size,s,:alloc,u,:bfree,f)); end
    end
    if n < k; println('⋮'); end
    println((:totbytes,bytes,:bfree,bfree,:ptrs,ptrs))
    if KnetMems != nothing
        mem = KnetMems[dev+1]
        println((:membytes,mem.bytes,:bfree,mem.bfree,:limit,mem.limit,:pools,length(mem.pools)))
    end
end

function memdbg(msg="")
    if nvmlfound # Libdl.find_library(["libnvidia-ml"],[]) != ""
        m = nvmlDeviceGetMemoryInfo()
    else
        m = (0,0,0)
    end
    c = cudaMemGetInfo()
    x = [0,0]
    for (s,u,f) in meminfo()
        x[1] += s*u
        x[2] += s*f
    end
    println("""$msg:
cudaMemGetInfo: ctotal: $(c[2]) cfree: $(c[1]) ctotal-cfree: $(c[2]-c[1])
nvmlDeviceGetMemoryInfo: ntotal: $(m[1]) nfree: $(m[2]) nused: $(m[3]) nfree+used: $(m[2]+m[3])
KnetPtr: ktotal: $(x[1]) kavail: $(x[2]) ktotal-kavail: $(x[1]-x[2])
nused-ktotal: $(m[3]-x[1])
""")
end

