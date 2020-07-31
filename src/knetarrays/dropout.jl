import ..Ops20: dropout!

for S in (32,64)
    T = Symbol("Float$S")
    forw = Symbol("dropout_$S")
    back = Symbol("dropback_$S")
    @eval begin
        function dropout!(p::Number, x::KnetArray{$T}, y::KnetArray{$T})
            rand!(y)
            @knet8($forw,(Cint,$T,Ptr{$T},Ptr{$T}),length(y),$T(p),x,y)
            return y
        end
        function dropback!(p::Number, x::KnetArray{$T}, y::KnetArray{$T}, dy::KnetArray{$T}, dx::KnetArray{$T})
            @knet8($back,(Cint,$T,Ptr{$T},Ptr{$T},Ptr{$T},Ptr{$T}),length(dx),$T(p),x,y,dy,dx)
            return dx
        end
    end
end
