using Knet, CuArrays

pd_knet(x::KnetArray, perm) = permutedims(x,perm)
pd_cpux(x::KnetArray, perm) = KnetArray(permutedims(Array(x),perm))
pd_cuxx(x::KnetArray, perm) = KnetArray(permutedims(CuArray(x),perm))
pd_kern(x::KnetArray, perm) = nothing # call specific kernel
pd_tran(x::KnetArray, perm) = transpose(x)

function CuArrays.CuArray(x::KnetArray{T}) where {T} # should we define convert instead?
    p = CuArrays.CuPtr{T}(UInt(x.ptr.ptr))  # This does not take into account shared subarrays?
    Base.unsafe_wrap(CuArray{T}, p, size(x); own=false)
end

function Knet.KnetArray(x::CuArray{T,N}) where {T,N}
    p = Base.bitcast(Knet.Cptr, x.buf.ptr)
    k = Knet.KnetPtr(p, sizeof(x), gpu(), x) 
    # finalizer(identity, k) # hacky way to avoid gc? gives error in running finalizer
    KnetArray{T,N}(k, size(x))
end

function pd_time(f, x, p, a=nothing, n=max(10,div(1000000,length(x))))
    println(f,size(x),p,n)
    y = f(x,p)
    if a != nothing; @assert y == a; end
    @time begin
        for i in 1:n
            f(x,p)
        end
        Knet.cudaDeviceSynchronize()
    end
    @time begin
        for i in 1:n
            f(x,p)
        end
        Knet.cudaDeviceSynchronize()
    end
    return y
end

@info "Simple transpose"
x = KnetArray(rand(Float32,100,100))
p = (2,1)
a = pd_time(pd_cpux, x, p)
pd_time(pd_tran, x, p, a)
pd_time(pd_knet, x, p, a) # confirm this uses transpose
pd_time(pd_cuxx, x, p, a)
#pd_time(pd_kern, x, p)
x = a = nothing; Knet.gc(); #Knet.gpuinfo()

@info "ILKER-DRAW"
x = KnetArray(rand(Float32,28,5,100))
p = (2,1,3)
a = pd_time(pd_cpux, x, p)
pd_time(pd_knet, x, p, a)
pd_time(pd_cuxx, x, p, a)
#pd_time(pd_kern, x, p, a)
x = a = nothing; Knet.gc(); #Knet.gpuinfo()

@info "ILKER-ATTN"
x = KnetArray(rand(Float32,196,512,32))
p = (2,1,3)
a = pd_time(pd_cpux, x, p)
pd_time(pd_knet, x, p, a)
pd_time(pd_cuxx, x, p, a)
#pd_time(pd_kern, x, p, a)
x = a = nothing; Knet.gc(); #Knet.gpuinfo()

@info "OSMAN-BERT-1"
x = KnetArray(rand(Float32,768,512,16))
p = (2,1,3)
a = pd_time(pd_cpux, x, p)
pd_time(pd_knet, x, p, a)
pd_time(pd_cuxx, x, p, a)
#pd_time(pd_kern, x, p, a)
x = a = nothing; Knet.gc(); #Knet.gpuinfo()

@info "OSMAN-BERT-2"
x = KnetArray(rand(Float32,64,512,12,16))
p = (1,3,2,4)
a = pd_time(pd_cpux, x, p)
pd_time(pd_knet, x, p, a)
pd_time(pd_cuxx, x, p, a)
#pd_time(pd_kern, x, p, a)
x = a = nothing; Knet.gc(); #Knet.gpuinfo()

nothing
