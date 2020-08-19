import Base: getindex, size, summary, show
using Base: dims2string
using AutoGrad: Value

# https://docs.julialang.org/en/v1/manual/types/#man-custom-pretty-printing-1
# show(io::IO, z): single line format used in show, print, inside other objects.
# show(io::IO, ::MIME"text/plain", z): multi-line format used by display.
# show(io::IO, ::MIME"text/html", z): multi-line format for html output.
# get(io, :compact, false), show(IOContext(stdout, :compact=>true),z) for compact (array) printing.
# summary(io::IO, x) = print(io, typeof(x))
# string(z): uses print_to_string.


# Hack for printing without copying the whole KnetArray and without inheriting AbstractArray:
struct KnetDisplay{T,N} <: AbstractArray{T,N}; a::KnetArray{T,N}; end
getindex(a::KnetDisplay, i...) = getindex(a.a, i...)
size(a::KnetDisplay) = size(a.a)
summary(io::IO, a::KnetDisplay) = summary(io, a.a)
summary(io::IO, a::KnetArray) = print(io, dims2string(size(a)), " ", typeof(a))
show(io::IO, m::MIME"text/plain", a::KnetArray) = show(io, m, KnetDisplay(a))
summary(io::IO, x::Value{A}) where {A<:KnetArray} = print(io, dims2string(size(x)), " ", typeof(x))

function show(io::IO, a::KnetArray) # Compact display used by print
    T = eltype(a)
    print(io, T <: AbstractFloat ? "K$(sizeof(T)*8)" : "K{$T}")
    print(io, "($(join(size(a),',')))")
    print(io, isempty(a) ? "[]" : "[$(a[1])⋯]")
end

