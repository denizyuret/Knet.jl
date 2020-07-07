using Knet, CUDA

pd_knet(y::KnetArray, x::KnetArray, perm) = permutedims!(y,x,perm)
pd_cpux(y::KnetArray, x::KnetArray, perm) = copyto!(y, permutedims(Array(x),perm))
pd_cuxx(y::KnetArray, x::KnetArray, perm) = (permutedims!(cu(y),cu(x),perm); y)
pd_kern(y::KnetArray, x::KnetArray, perm) = nothing # call specific kernel
pd_tran(y::KnetArray, x::KnetArray, perm) = Knet._transpose!(y,x)

function CUDA.cu(x::KnetArray{T}) where {T}
    p = CUDA.CuPtr{T}(UInt(x.ptr.ptr))
    Base.unsafe_wrap(CuArray{T}, p, size(x); own=false)
end

# ## best not to use CUDA.jl memory manager
# function Knet.ka(x::CuArray{T,N}) where {T,N}
#     p = Base.bitcast(Knet.Cptr, x.buf.ptr)
#     k = Knet.KnetPtr(p, sizeof(x), gpu(), x) 
#     # finalizer(identity, k) # hacky way to avoid gc? gives error in running finalizer
#     KnetArray{T,N}(k, size(x))
# end

function pd_time(f, x, p, a=nothing, n=max(10,div(1000000,length(x))))
    println(f,size(x),p,n)
    y = similar(x, size(x)[[p...]])
    f(y,x,p)
    if a != nothing; @assert y == a; end
    @time begin
        for i in 1:n
            f(y,x,p)
        end
        Knet.cudaDeviceSynchronize()
    end
    @time begin
        for i in 1:n
            f(y,x,p)
        end
        Knet.cudaDeviceSynchronize()
    end
    return y
end

@info "Simple transpose"
p = (2,1)
x = KnetArray(rand(Float32,100,100))
a = pd_time(pd_cpux, x, p)
pd_time(pd_tran, x, p, a)
pd_time(pd_knet, x, p, a) # confirm this uses transpose
pd_time(pd_cuxx, x, p, a)
#pd_time(pd_kern, x, p)
x = y = a = nothing; Knet.gc(); #Knet.gpuinfo()

@info "ILKER-DRAW"
p = (2,1,3)
x = KnetArray(rand(Float32,28,5,100))
a = pd_time(pd_cpux, x, p)
pd_time(pd_knet, x, p, a)
pd_time(pd_cuxx, x, p, a)
#pd_time(pd_kern, x, p, a)
x = y = a = nothing; Knet.gc(); #Knet.gpuinfo()

@info "ILKER-ATTN"
p = (2,1,3)
x = KnetArray(rand(Float32,196,512,32))
a = pd_time(pd_cpux, x, p)
pd_time(pd_knet, x, p, a)
pd_time(pd_cuxx, x, p, a)
#pd_time(pd_kern, x, p, a)
x = y = a = nothing; Knet.gc(); #Knet.gpuinfo()

@info "OSMAN-BERT-1"
p = (2,1,3)
x = KnetArray(rand(Float32,768,512,16))
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
