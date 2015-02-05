### TEST CODE

include("jnet.jl")

using MAT
using CUDArt

function load()
    file = matopen("dev.mat")
    data = read(file, "dev")
    close(file)
    return data
end

function test(data)
    info("Initializing...")
    x = data["x"]
    ylabels = vec(data["y"])
    y = full(sparse(convert(Vector{Int},ylabels), 1:length(ylabels), ones(ylabels)))
    w1 = data["w1"][:,2:end]
    b1 = data["w1"][:,1]
    w2 = data["w2"][:,2:end]
    b2 = data["w2"][:,1]

    device_reset(0)
    net = [Jnet.relu(w1,b1), Jnet.soft(w2,b2)]
    info("GPU Forward 1")
    @time y1 = Jnet.forward(net, x)
    info("GPU Forward 2")
    @time y2 = Jnet.forward(net, x)
    assert(y1==y2)
    
    (a,b) = findmax(y1,1)
    b = vec(b)
    broadcast!(mod1, b, b, 3)
    info("Accuracy=$(mean(b.==ylabels))")
    
    info("GPU Forwback 1")
    @time y1 = Jnet.forwback(net, x, y)
    info("GPU Forwback 2")
    @time y2 = Jnet.forwback(net, x, y)
    return net
end


#     l1 = ccall(("relu","./libjnet.so"), Ptr{Void}, (Cint,Cint,Ptr{Float32},Ptr{Float32}), size(w1,1), size(w1,2), w1, b1)
#     l2 = ccall(("soft","./libjnet.so"), Ptr{Void}, (Cint,Cint,Ptr{Float32},Ptr{Float32}), size(w2,1), size(w2,2), w2, b2)
#     n0 = [l1, l2]
#     info("CPU Forward 1")
#     y1 = similar(w1, size(w2,1), size(x10k, 2))
#     @time ccall(("forward","./libjnet.so"), Void, 
# 		(Ptr{Ptr{Void}}, Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint), 
# 		n0, x10k, y1, length(n0), size(x10k, 2), 100)
#     info("CPU Forward 2")
#     y2 = similar(w1, size(w2,1), size(x10k, 2))
#     @time ccall(("forward","./libjnet.so"), Void, 
# 		(Ptr{Ptr{Void}}, Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint), 
# 		n0, x10k, y2, length(n0), size(x10k, 2), 100)
#     assert(y1==y2)
#     ccall(("layer_free","./libjnet.so"), Void, (Ptr{Void},), l1)
#     ccall(("layer_free","./libjnet.so"), Void, (Ptr{Void},), l2)
#     return y1
# end
    
#     n0 = [relu(w1,b1), soft(w2,b2)]
#     info("Running 10k cpu test with bias...")
#     info("CPU Forward 1")
#     @time y1 = forw(n0, x10k, 100)
#     info("CPU Forward 2")
#     @time y2 = forw(n0, x10k, 100)
#     assert(y1 == y2)
#     info("CPU Forwback 1")
#     @time forwback(n0, x10k, y10k, 100)
#     info("CPU Forwback 2")
#     @time forwback(n0, x10k, y10k, 100)

#     CUDArt.device_reset(0)
#     gw1 = CudaArray(w1)
#     gb1 = CudaArray(b1)
#     gw2 = CudaArray(w2)
#     gb2 = CudaArray(b2)
#     g0 = [relu(gw1,gb1), soft(gw2,gb2)]
#     info("Running 10k gpu test with bias...")
#     info("GPU Forward 1")
#     @time y3 = forw(g0, x10k, 100)
# #    info("maxabsdiff=$(maximum(abs(y3-y2)))")
#     info("GPU Forward 2")
#     @time y4 = forw(g0, x10k, 100)
#     assert(y4 == y3)
#     info("GPU Forwback 1")
#     @time forwback(g0, x10k, y10k, 100)
#     info("GPU Forwback 2")
#     @time forwback(g0, x10k, y10k, 100)
#     info("done")
#     return g0
# #  end # devices

# end

# end # function test(data)

