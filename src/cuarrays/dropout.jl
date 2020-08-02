using CUDA, Random
import ..Ops20: dropout!, dropback!

# GPU implementation
for S in (32,64)
    T = Symbol("Float$S")
    forw = Symbol("dropout_$S")
    back = Symbol("dropback_$S")
    @eval begin
        function dropout!(p::Number, x::CuArray{$T}, y::CuArray{$T})
            rand!(y)
            @knet8($forw,(Cint,$T,CuPtr{$T},CuPtr{$T}),length(y),$T(p),x,y)
            return y
        end
        function dropback!(p::Number, x::CuArray{$T}, y::CuArray{$T}, dy::CuArray{$T}, dx::CuArray{$T})
            @knet8($back,(Cint,$T,CuPtr{$T},CuPtr{$T},CuPtr{$T},CuPtr{$T}),length(dx),$T(p),x,y,dy,dx)
            return dx
        end
    end
end
