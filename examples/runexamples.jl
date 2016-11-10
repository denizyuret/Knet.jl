using Knet

Pkg.test("Knet")
load_only = true

for (p,f,o1,o2,o3) =
    (
     (:LinReg, "linreg.jl", "--gcheck 2", "--fast", "--fast"),
     (:Housing, "housing.jl", "--gcheck 2 --atype KnetArray{Float64}", "--fast", "--fast"),
     (:MNIST, "mnist.jl", "--gcheck 2", "--fast", "--fast"),
     (:LeNet, "lenet.jl", "--gcheck 2", "--fast", "--fast"),
     (:CharLM, "charlm.jl", "--gcheck 2 --winit 0.01", "--fast", "--fast"),
    )
    gpu() < 0 && p == :LeNet && continue
    gc()
    Knet.knetgc()
    include(f)
    m = eval(:($p.main))
    m(o1); m(o2); m(o3)
end
