using Knet

Pkg.test("Knet")
load_only = true

for (p,f,o1,o2,o3) =
    (
     (:LinReg, "linreg.jl", "--gcheck 2", "--fast", "--fast"),
     (:Housing, "housing.jl", "--gcheck 2", "--fast", "--fast"),
     (:MNIST, "mnist.jl", "--gcheck 2", "--fast", "--fast"),
     (:LeNet, "lenet.jl", "--gcheck 2", "--fast", "--fast"),
     (:CharLM, "charlm.jl", "--gcheck 2 --data 10.txt", "--fast --data 10.txt", "--fast --data 10.txt"),
    )
    gc()
    Knet.knetgc()
    include(f)
    m = eval(:($p.main))
    m(o1); m(o2); m(o3)
end

# include("linreg.jl"); LinReg.main("--gcheck 2"); LinReg.main("--fast"); LinReg.main("--fast")
# include("housing.jl"); Housing.main("--gcheck 2"); Housing.main("--fast"); Housing.main("--fast")
# include("mnist.jl"); MNIST.main("--gcheck 2"); MNIST.main("--fast"); MNIST.main("--fast")
# include("lenet.jl"); LeNet.main("--gcheck 2"); LeNet.main("--fast"); LeNet.main("--fast")
# include("charlm.jl"); CharLM.main("--gcheck 2 --data 10.txt"); CharLM.main("--data 10.txt"); CharLM.main("--data 10.txt")


# function predict1(w,x0)                       # 28,28,1,100
#     x1 = pool4(relu(conv4(w[1],x0) .+ w[2])) # 12,12,20,100
# end
# function predict2(w,x0)                       # 28,28,1,100
#     x1 = pool4(relu(conv4(w[1],x0) .+ w[2])) # 12,12,20,100
#     x2 = pool4(relu(conv4(w[3],x1) .+ w[4])) # 4,4,50,100
# end
# function predict3(w,x0)                       # 28,28,1,100
#     x1 = pool4(relu(conv4(w[1],x0) .+ w[2])) # 12,12,20,100
#     x2 = pool4(relu(conv4(w[3],x1) .+ w[4])) # 4,4,50,100
#     x2a = reshape(x2, (800,100))             # 800,100
# end
# function predict4(w,x0)                       # 28,28,1,100
#     x1 = pool4(relu(conv4(w[1],x0) .+ w[2])) # 12,12,20,100
#     x2 = pool4(relu(conv4(w[3],x1) .+ w[4])) # 4,4,50,100
#     x2a = reshape(x2, (800,100))             # 800,100
#     x3 = relu(w[5]*x2a .+ w[6])              # 500,100
# end
# function predict5(w,x0)                       # 28,28,1,100
#     x1 = pool4(relu(conv4(w[1],x0) .+ w[2])) # 12,12,20,100
#     x2 = pool4(relu(conv4(w[3],x1) .+ w[4])) # 4,4,50,100
#     x2a = reshape(x2, (800,100))             # 800,100
#     x3 = relu(w[5]*x2a .+ w[6])              # 500,100
#     x4 = w[7]*x3 .+ w[8]                     # 10,100
# end

# a = KnetArray(rand(3,5))
# b = a[3:5]

# function f2()
#     a = Any[]
#     for i=1:10
#         push!(a, KnetArray(Float32, 1000, 1000))
#     end
#     gpuinfo()
#     empty!(a); a=nothing; gc(); sleep(1)
#     gpuinfo()
#     gc(); sleep(1)
#     gpuinfo()
# end

# function f1(x)
#     s = 0
#     for i = 1:length(x)
#         if x[i] <= 0
#             s += exp(x[i])
#         else
#             s += log(x[i])
#         end
#     end
#     return s
# end
