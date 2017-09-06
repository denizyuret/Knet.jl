#          osx4 osx5 ai4  ai5
# linreg   1.03 1.51 0.94 1.08
# housing  0.01 0.01 0.01 0.01
# mnist    4.66 4.96 1.57 1.73
# lenet    -    -    6.55 7.18
# charlm   20.0 22.7 0.33 0.34

using Knet

# Pkg.test("Knet")
load_only = true

for (p,f,o1,o2,o3) =
    (
     (:LinReg, "linreg.jl", "--gcheck 2", "--fast", "--fast"),
     (:Housing, "housing.jl", "--gcheck 2 --atype Array{Float64}", "--fast", "--fast"),
     (:MNIST, "mnist.jl", "--gcheck 2", "--fast", "--fast"),
     (:LeNet, "lenet.jl", "--gcheck 2", "--fast", "--fast"),
     #(:CharLM, "charlm.jl", "--gcheck 2 --winit 0.01", "--fast", "--fast"),
     # TODO: gcheck errors for charlm
     (:CharLM, "charlm.jl", "--gcheck 2", "--fast", "--fast"),
    )
    gpu() < 0 && p == :LeNet && continue
    include(f)
    m = eval(:($p.main))
    m(o1); m(o2); m(o3)
    gc(); gpu() >= 0 && Knet.knetgc()
end
