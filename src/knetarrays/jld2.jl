import JLD2, FileIO

struct JLD2KnetArray{T,N}; array::Array{T,N}; end
JLD2.writeas(::Type{KnetArray{T,N}}) where {T,N} = JLD2KnetArray{T,N}
JLD2.wconvert(::Type{JLD2KnetArray{T,N}}, x::KnetArray{T,N}) where {T,N} = JLD2KnetArray(Array(x))
JLD2.rconvert(::Type{KnetArray{T,N}}, x::JLD2KnetArray{T,N}) where {T,N} = KnetArray(x.array)

# Deprecate these eventually
load(file, args...; options...) = FileIO.load(file, args...; options...)
save(file, args...; options...) = FileIO.save(file, args...; options...)
macro save(file, args...); ex=:(JLD2.@save($file)); append!(ex.args,args); ex; end
macro load(file, args...); ex=:(JLD2.@load($file)); append!(ex.args,args); ex; end
