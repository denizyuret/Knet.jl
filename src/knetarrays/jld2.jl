import JLD2, FileIO

# With the following standard FileIO.save, FileIO.load, JLD2.@save, JLD2.@load should work
struct JLD2KnetArray{T,N}; array::Array{T,N}; end
JLD2.writeas(::Type{KnetArray{T,N}}) where {T,N} = JLD2KnetArray{T,N}
JLD2.wconvert(::Type{JLD2KnetArray{T,N}}, x::KnetArray{T,N}) where {T,N} = JLD2KnetArray(Array(x))
JLD2.rconvert(::Type{KnetArray{T,N}}, x::JLD2KnetArray{T,N}) where {T,N} = KnetArray(x.array)

# For people who use Knet.load, Knet.@save etc.
save(x...; o...) = FileIO.save(x...; o...)
load(x...; o...) = FileIO.load(x...; o...)
macro save(x...); a=:(JLD2.@save); append!(a.args,x); a; end
macro load(x...); a=:(JLD2.@load); append!(a.args,x); a; end

# To load a file saved with Knet-1.4.3 or earlier:
load143(x...; o...)=jld2serialize(FileIO.load(x...; o...))
