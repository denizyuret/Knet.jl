"""
@knet Op nce(q,s) computes the activation function for Noise
Contrastive Estimation.  Given q[K] and s[K,B] the output is p[K,B]
where p[i,j]=exp(s[i,j])/(exp(s[i,j])+q[i]).
"""
type NCE <: Op; NCE(;o...)=new(); end

ninputs(::NCE)=2
canoverwrite(::NCE)=true
back_reads_x(::NCE)=false
back_reads_y(::NCE)=true

# kq[K,1], s[K,B], p[i,j]=exp(s[i,j])/(exp(s[i,j])+kq[i])
nceforw!(kq::Array,s::Array,p::Array)=(for i=1:size2(s,1), j=1:size2(s,2); p[i,j]=exp(s[i,j])/(exp(s[i,j])+kq[i]); end; p)

@gpu nceforw!{T}(kq::CudaArray{T},s::CudaArray{T},p::CudaArray{T})=
    (T <: Float32 ? ccall((:nceforw32,libknet),Void,(Cint,Cint,Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}), size2(s,1),size2(s,2),kq,s,p) :
     T <: Float64 ? ccall((:nceforw64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),size2(s,1),size2(s,2),kq,s,p) :
     error("$T not supported"); gpusync(); p)

forw(::NCE, kq, s, p; o...)=(size2(s) != size2(p) ? throw(DimensionMismatch()) :
                             size2(s,1) != length(kq) ? throw(DimensionMismatch()) :
                             nceforw!(kq,s,p))

# ds[i,j] = dp[i,j]*p[i,j]*(1-p[i,j]) same as sigmback!
function back(::NCE, dp, dq, ds; y=nothing, o...) # y is p
    dq != nothing && (Base.warn_once("Taking gradient of constant"); fillsync!(dq,0))
    ds != nothing && (length(y)==length(ds)==length(dp)||throw(DimensionMismatch()); sigmback(y,dp,ds))
end

function infersize(a::NCE, q, s, p)
    q = (q == nothing ? [0,1] : length(q) == 2 ? [q...] : throw(DimensionMismatch("q=$q")))
    s = (s == nothing ? [0,0] : length(s) == 2 ? [s...] : throw(DimensionMismatch("s=$s")))
    p = (p == nothing ? [0,0] : length(p) == 2 ? [p...] : throw(DimensionMismatch("p=$p")))
    q[2] = (q[2]==1 ? 1 : q[2]==0 ? 1 : throw(DimensionMismatch("q=$q")))
    p1 = (q[1] > 0 ? q[1] : s[1] > 0 ? s[1] : p[1])
    p2 = (s[2] > 0 ? s[2] : p[2])
    if p1 > 0
        q[1]==p1 ? nothing : q[1]==0 ? q[1]=p1 : throw(DimensionMismatch())
        s[1]==p1 ? nothing : s[1]==0 ? s[1]=p1 : throw(DimensionMismatch())
        p[1]==p1 ? nothing : p[1]==0 ? p[1]=p1 : throw(DimensionMismatch())
    end
    if p2 > 0
        s[2]==p2 ? nothing : s[2]==0 ? s[2]=p2 : throw(DimensionMismatch())
        p[2]==p2 ? nothing : p[2]==0 ? p[2]=p2 : throw(DimensionMismatch())
    end
    return (tuple(q...), tuple(s...), tuple(p...))
end


### DEAD CODE:
# nce(q,s,p)=(NCE(),q,s,p)
