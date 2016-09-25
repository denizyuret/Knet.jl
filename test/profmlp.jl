isdefined(:MNIST) || include(Pkg.dir("Knet/examples/mnist.jl"))
using Knet, BenchmarkTools
using MNIST: weights, minibatch, xtrn, ytrn
using Knet: gpusync

atype  = KnetArray{Float32}
hidden = [64]
batch  = 100
data   = minibatch(xtrn, ytrn, batch, atype=atype)
model  = weights(64, atype=atype)

prog = quote
    a1 = w[1]*x                 # 0.67
    a2 = w[2].+a1               # 0.71
    a3 = relu(a2)               # 0.75 => 2.75-2.77
    #a3 = max(0,a2)              # 0.75 => 2.79-2.84
    a4 = w[3]*a3                # 0.81
    a5 = w[4].+a4               # 0.85
    a6 = y - a5                 # 0.89
    #a6 = logp(a5,1)             # 1.07 => 3.31-3.41
    #a6 = a5 .- log(sum(exp(a5),1)) # 0.99 => 3.56-3.67
    #a7 = a6.*a6                 # 0.92 => 3.02-3.06
    #a7 = a6.^2                  # 0.92 => 2.93-3.00
    #a7 = abs2(a6)               # 0.92 => 2.93-3.00
    #a7 = y .* a6                # 1.10 => 3.31-3.41
    a7 = sumabs2(a6)            # 1.18 => 1.32-1.35,2.79-2.84
    #a8 = sum(a7)                # 1.22 => 2.93-3.00
end

prog = filter(p->isa(p,Expr), prog.args)
fun = Any[]

for i=1:length(prog)
    f = Symbol("f$i")
    eval(Expr(:function, :($f(w,x,y)), Expr(:block,prog[1:i]...)))
    push!(fun, eval(f))
end

ffin = fun[end]
push!(fun,(w,x,y)->AutoGrad.forward_pass(ffin,(w,x,y),(),1))
push!(fun,grad(ffin))

function loop(f,w,d,t)
    for i in 1:t
        for (x,y) in d
            f(w,x,y)
        end
    end
end

@time for i=1:length(fun)
    loop(fun[i],model,data,1)   # to compile
end

function profmlp(fun,model,data,range=1:length(fun); samples=5, seconds=30, o...)
    for i in range
        f = fun[i]
        b = @benchmarkable (loop($f,$model,$data,10);gpusync())
        t = run(b; samples=samples, seconds=seconds, o...).times ./ 10^9
        println((:f,i,:n,length(t),:min,minimum(t),:med,median(t),:avg,mean(t),:max,maximum(t)))
    end
end

# (:f,1,:n,5,:min,0.674071033,:med,0.674303586,:avg,0.6742660326000001,:max,0.674379312)
# (:f,2,:n,5,:min,0.71386119,:med,0.714166844,:avg,0.714133768,:max,0.714420962)
# (:f,3,:n,5,:min,0.749161714,:med,0.749578344,:avg,0.7576167076000001,:max,0.789836767)
# (:f,4,:n,5,:min,0.812147525,:med,0.812692507,:avg,0.81255644,:max,0.812902328)
# (:f,5,:n,5,:min,0.850032182,:med,0.850729729,:avg,0.8505739558000001,:max,0.850794662)
# (:f,6,:n,5,:min,0.889973775,:med,0.890434318,:avg,0.8905268512,:max,0.890990265)
# (:f,7,:n,5,:min,0.916586753,:med,0.91684639,:avg,0.9174720356,:max,0.919871972)
# (:f,8,:n,5,:min,1.218039367,:med,1.223258585,:avg,1.222431379,:max,1.226436017)
# (:f,9,:n,5,:min,1.433074136,:med,1.437228564,:avg,1.4373717761999998,:max,1.441818492)
# (:f,10,:n,5,:min,2.967745663,:med,2.975174783,:avg,2.9851800988,:max,3.029562013)
