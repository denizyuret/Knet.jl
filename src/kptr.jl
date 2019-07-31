# KnetPtr type holds a gpu allocated pointer.  We try to minimize the number of actual
# allocations, which are slow, by reusing preallocated but garbage collected pointers.

mutable struct KnetPtr
    ptr                         # actual pointer, removed type ::Cptr for serialization
    len::Int                    # size in bytes
    dev::Int                    # id of the device the pointer belongs to
    parent                      # used to implement shared memory pointers

    # This is the low level KnetPtr constructor, it adds the finalizer and
    # does not assign parent which is only needed for shared pointers.

    function KnetPtr(ptr::Cptr,len::Int,dev::Int,parent=nothing)
        kp = new(ptr,len,dev,parent)
        if !isa(parent,Nothing); finalizer(freeKnetPtr, kp); end
        return kp
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
    if p.ptr == C_NULL || !isa(p.parent,Nothing); return; end
    mem = KnetMems[p.dev+1]
    mem.avail += p.len
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
    avail::Int                  # total bytes freed and available
    KnetMem()=new(Dict{Int,KnetPool}(),KNETMEMINIT,0,0)
end

# KnetMems[dev+1] holds memory information for device dev.
KnetMems = nothing
initKnetMems() = (global KnetMems = [ KnetMem() for i in 1:gpuCount() ])

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

function inclimit!(m::KnetMem, minlimit=m.limit*6÷5)
    maxlimit = gpumem()[1] - 500_000_000 # m.bytes + gpufree()
    minlimit = min(maxlimit, minlimit)
    if m.limit < minlimit
        m.limit = minlimit
        @debug "limit=$(m.limit) bytes=$(m.bytes) avail=$(m.avail) pools=$(length(m.pools)) blocks=$(sum(p->p.nptr,values(m.pools)))"
    end
end

# This the main KnetPtr constructor.  It tries to avoid actual
# allocation which is slow.  Reusing a pointer is very fast.
# Allocating a new one is about 10 μs.  gc() is about 75 ms.

function KnetPtr(arraybytes::Int)
    dev = gpu(); if dev < 0; error("KnetPtr: bad device id $dev."); end
    if KnetMems==nothing; initKnetMems(); end
    mem = KnetMems[dev+1]
    blockbytes = blocksize(arraybytes)
    @dbg (push!(arraysizes,arraybytes); push!(blocksizes,blockbytes))
    pool = get!(KnetPool,mem.pools,blockbytes)
    if !isempty(pool.free)      # 1. best case we have one available in pool
        @dbg push!(allocs, 1)
        mem.avail -= blockbytes
        return KnetPtr(pop!(pool.free),blockbytes,dev)
    end
    if mem.bytes + blockbytes <= mem.limit # 2. allocate if we are within limit
        ptr = knetMalloc(blockbytes)
        if ptr != nothing
            @dbg push!(allocs, 2)
            mem.bytes += blockbytes
            pool.nptr += 1
            return KnetPtr(ptr,blockbytes,dev)
        end
    end
    if pool.nptr > 0            # 3. try gc if we had allocated this size before
        GC.gc();  @dbg print('-')
        # inclimit!(mem, (mem.bytes-mem.avail)*3) # *4=>18s *3=>16s *5÷2=>16s *2=>22s *6÷5=>30s
        inclimit!(mem, mem.limit*6÷5)
        if !isempty(pool.free)
            @dbg push!(allocs, 3)
            mem.avail -= blockbytes
            return KnetPtr(pop!(pool.free),blockbytes,dev)
        end
    end
    # At this point we have to either free old ptrs or increase limit
    # Let's just try limit increase for now
    inclimit!(mem, (mem.bytes + blockbytes*2))
    ptr = knetMalloc(blockbytes)
    if ptr == nothing
        Knet.gc(); @dbg print('+')  # last ditch effort
        ptr = knetMalloc(blockbytes)
    end
    if ptr != nothing
        @dbg push!(allocs, 4)
        mem.bytes += blockbytes
        pool.nptr += 1
        return KnetPtr(ptr,blockbytes,dev)
    end
    error("Out of gpu memory")
end

KnetPtr(n::Integer)=KnetPtr(Int(n))

# This does the actual allocation, returns `nothing` in case of error
function knetMalloc(nbytes::Int)
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
    mem = KnetMems[dev+1]
    GC.gc(); GC.enable(false)
    for (n,v) in mem.pools
        for p in v.free
            @cudart(cudaFree,(Cptr,),p)
        end
        v.nptr -= length(v.free)
        mem.avail -= n * length(v.free)
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
    if KnetMems != nothing
        mem = KnetMems[dev+1]
        println((:membytes,mem.bytes,:avail,mem.avail,:limit,mem.limit,:pools,length(mem.pools)))
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
