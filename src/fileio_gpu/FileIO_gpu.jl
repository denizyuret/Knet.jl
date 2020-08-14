module FileIO_gpu

import FileIO # save, load
using JLD2: JLD2, JLDWriteSession, jldopen, isgroup, lookup_offset
using Knet.KnetArrays: KnetPtr, KnetArray
using Knet.Ops20: RNN

include("serialize.jl"); export cpucopy, gpucopy
include("jld.jl"); export save, load, @save, @load

end
