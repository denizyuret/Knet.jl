### TEST CODE

using MAT

function load()
    file = matopen("dev.mat")
    data = read(file, "dev")
    close(file)
    return data
end

using Jnet

function test(data)
    # result = devices((x->true), nmax=1) do devlist
    # device(devlist[1])

    # info("Loading...")
    # data = load()

    info("Initializing...")
    x10k = data["x"][:,1:10000]
    y10k = data["y"][1:10000]
    w1 = data["w1"][:,2:end]
    b1 = data["w1"][:,1]
    w2 = data["w2"][:,2:end]
    b2 = data["w2"][:,1]

if false
    n0 = [relu(w1,b1), soft(w2,b2)]
    info("Running 10k cpu test with bias...")
    info("CPU Forward 1")
    @time y1 = forw(n0, x10k, 100)
    info("CPU Forward 2")
    @time y2 = forw(n0, x10k, 100)
    assert(y1 == y2)
    info("CPU Forwback 1")
    @time forwback(n0, x10k, y10k, 100)
    info("CPU Forwback 2")
    @time forwback(n0, x10k, y10k, 100)
end

    CUDArt.device_reset(0)
    gw1 = CudaArray(w1)
    gb1 = CudaArray(b1)
    gw2 = CudaArray(w2)
    gb2 = CudaArray(b2)
    g0 = [relu(gw1,gb1), soft(gw2,gb2)]
    info("Running 10k gpu test with bias...")
    info("GPU Forward 1")
    @time y3 = forw(g0, x10k, 100)
#    info("maxabsdiff=$(maximum(abs(y3-y2)))")
    info("GPU Forward 2")
    @time y4 = forw(g0, x10k, 100)
    assert(y4 == y3)
    info("GPU Forwback 1")
    @time forwback(g0, x10k, y10k, 100)
    info("GPU Forwback 2")
    @time forwback(g0, x10k, y10k, 100)
    info("done")
    return g0
#  end # devices


end # function test(data)

