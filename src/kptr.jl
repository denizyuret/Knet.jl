GCDBG=true; macro gcdbg(expr); if GCDBG; esc(expr); end; end

# KnetPtr type holds a gpu allocated pointer.  We try to minimize the number of actual
# allocations, which are slow, by reusing preallocated but garbage collected pointers.

mutable struct KnetPtr
    ptr::Cptr                   # actual pointer
    len::Int                    # size in bytes
    dev::Int                    # id of the device the pointer belongs to
    parent::KnetPtr             # used to implement shared memory pointers

    # This is the low level KnetPtr constructor, it adds the finalizer and
    # sets parent to `nothing` which is only needed for shared pointers.

    function KnetPtr(ptr::Cptr,len::Int,dev::Int)
        kp = new(ptr,len,dev)
        finalizer(freeKnetPtr, kp)
    end

    # This constructor is used to create a shared pointer.  We need to
    # keep the parent field to prevent premature gc of the parent.  The
    # child does not need a special finalizer.

    function KnetPtr(parent::KnetPtr, offs::Integer, len::Integer)
        if len < 0 || offs < 1 || offs+len-1 > parent.len; throw(BoundsError()); end
        new(parent.ptr+offs-1, len, parent.dev, parent)
    end

end

# When Julia gc reclaims a KnetPtr object, the following special finalizer does not actually
# release the memory, but inserts it back in the appropriate pool for reuse.

function freeKnetPtr(p::KnetPtr)
    mem = KnetMems[p.dev+1]
    mem.avail += p.len
    push!(mem.pools[p.len].free, p.ptr)
end

# We use the KnetPool type to keep track of allocated and garbage collected pointers: We
# keep one KnetPool struct per size per device.

mutable struct KnetPool
    free::Vector{Cptr}          # pointers available for reuse
    nptr::Int                   # number of allocated pointers
    last::Int                   # last time this pool was used
    KnetPool()=new(Cptr[],0,0)
end

const KNETMEMLIMIT = 1<<30      # increase limit 1GB at a time

# KnetMem type keeps memory information for one device.
mutable struct KnetMem
    pools::Dict{Int,KnetPool}   # pointers of a given size
    limit::Int                  # current memory limit
    bytes::Int                  # total bytes allocated (inuse + avail)
    avail::Int                  # total bytes freed and available
    calls::Int                  # timer to measure pool age
    KnetMem()=new(Dict{Int,KnetPool}(),0,0,0,0)
end

# KnetMems[dev+1] holds memory information for device dev.
KnetMems = nothing
initKnetMems() = (global KnetMems = [ KnetMem() for i in 1:gpuCount() ])

# Blocksize determines the actual allocation size given the array size in bytes, and can be
# larger than what the array needs for increased reuse.
blocksize(n::Int)=n
# function blocksize(n::Int)
#     z = leading_zeros(n-1)
#     1<<(sizeof(n)*8-z)
# end

@gcdbg (arraysizes = Int[]; allocs = Int[]; blocksizes = Int[])

# This description is out-of-date, rewrite (TODO):
# This the main KnetPtr constructor.  It tries to avoid actual allocation which is slow.
# First it tries to find a previously allocated and garbage collected pointer in
# KnetFree[dev+1].  If not available it tries to allocate a new one (about 10 μs).
# Otherwise it tries running gc() and see if we get a pointer back (about 75 ms).  Finally
# if all else fails, it calls Knet.gc which cleans up all allocated and garbage collected
# KnetPool on the current device and tries allocation one last time.

function trypool(mem::KnetMem, pool::KnetPool, blockbytes::Int, dev::Int, a::Int=0)
    if !isempty(pool.free)
        @gcdbg push!(allocs, a+1)
        mem.avail -= blockbytes
        return KnetPtr(pop!(pool.free),blockbytes,dev)
    end
    if mem.bytes + blockbytes <= mem.limit
        ptr = knetMalloc(blockbytes)
        if ptr != nothing
            @gcdbg push!(allocs, a+2)
            mem.bytes += blockbytes
            pool.nptr += 1
            return KnetPtr(ptr,blockbytes,dev)
        end
    end
end

function KnetPtr(arraybytes::Int)
    if KnetMems==nothing; initKnetMems(); end
    dev = gpu(); if dev < 0; error("KnetPtr: bad device id $dev."); end
    mem = KnetMems[dev+1]
    mem.calls += 1
    blockbytes = blocksize(arraybytes)
    @gcdbg (push!(arraysizes,arraybytes); push!(blocksizes,blockbytes))
    pool = get!(KnetPool,mem.pools,blockbytes)
    pool.last = mem.calls
    ptr = trypool(mem,pool,blockbytes,dev)
    if ptr != nothing; return ptr; end
    GC.gc();  @gcdbg print('-')
    memfree = gpufree(dev)
    if blockbytes > mem.avail + memfree; error("Not enough space for $blockbytes bytes"); end
    maxlimit = mem.bytes + memfree
    if mem.avail < (KNETMEMLIMIT >> 1)
        mem.limit = min(maxlimit, mem.limit + KNETMEMLIMIT)
        @gcdbg (@show mem.limit, mem.bytes, mem.avail)
    end
    while mem.limit < maxlimit && mem.avail + mem.limit - mem.bytes < blockbytes
        mem.limit = min(maxlimit, mem.limit + KNETMEMLIMIT)
        @gcdbg (@show mem.limit, mem.bytes, mem.avail)
    end
    ptr = trypool(mem,pool,blockbytes,dev,2)
    if ptr != nothing; return ptr; end
    Knet.gc(dev, mem.bytes + blockbytes - mem.limit); @gcdbg print('+')
    ptr = trypool(mem,pool,blockbytes,dev,4)
    if ptr != nothing; return ptr; end
    error("Out of gpu memory")
end

KnetPtr(n::Integer)=KnetPtr(Int(n))

# This does the actual allocation, returns `nothing` in case of error
function knetMalloc(nbytes::Int)
    ptr = Cptr[0]
    ret = @cuda1(cudart,cudaMalloc,(Ptr{Cptr},Csize_t),ptr,nbytes)
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
    mem = KnetMems[dev+1]
    GC.gc(); GC.enable(false)
    for (n,v) in mem.pools
        for p in v.free
            @cuda(cudart,cudaFree,(Cptr,),p)
        end
        v.nptr -= length(v.free)
        mem.avail -= n * length(v.free)
        mem.bytes -= n * length(v.free)
        empty!(v.free)
    end
    GC.enable(true); GC.gc()
end

# this version cudaFree's n bytes from oldest pointers (TODO: do this faster with a heap)
function gc(dev::Int, nbytes::Int)
    mem = KnetMems[dev+1]
    pools = sort(collect(mem.pools),by=(x->x[2].last))
    freed = 0
    for (n,v) in pools
        @gcdbg (!isempty(v.free) && println("pool($n): free=$(length(v.free)) nptr=$(v.nptr) last=$(v.last) memcalls=$(mem.calls)"))
        while !isempty(v.free)
            p = pop!(v.free)
            @cuda(cudart,cudaFree,(Cptr,),p)
            v.nptr -= 1
            mem.avail -= n
            mem.bytes -= n
            freed += n
            if freed >= nbytes; return; end
        end
    end
    error("Knet.gc could not free $nbytes bytes.")
end

@deprecate knetgc Knet.gc

# Some utilities
meminfo(i=gpu())=(KnetMems==nothing ? [] : [(k,v.nptr,length(v.free)) for (k,v) in KnetMems[i+1].pools])

function gpuinfo(msg="",dev=gpu();n=10)
    msg != "" && print("$msg ")
    if nvmlfound # Libdl.find_library(["libnvidia-ml"],[]) != ""
        g = nvmlDeviceGetMemoryInfo(dev)
        println((:dev,dev,:total,g[1],:free,g[2],:used,g[3]))
    else
        dev0 = gpu(); gpu(dev)
        (free,total) = cudaMemGetInfo()
        gpu(dev0)
        println((:dev,dev,:total,total,:free,free,:used,total-free))
    end
    bytes = avail = ptrs = k = 0
    for (s,u,f) in sort(meminfo(dev), by=(x->x[1]*x[2]), rev=true)
        bytes += s*u; avail += s*f; ptrs += u; k += 1
        if k <= n; println((:bytes,s*u,:size,s,:alloc,u,:avail,f)); end
    end
    if n < k; println('⋮'); end
    println((:totbytes,bytes,:avail,avail,:ptrs,ptrs))
    mem = KnetMems[dev+1]
    println((:membytes,mem.bytes,:avail,mem.avail,:limit,mem.limit,:calls,mem.calls,:pools,length(mem.pools)))
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
