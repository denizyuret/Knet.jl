isdefined(:MNIST) || include(Pkg.dir("Knet/examples/mnist.jl"))
using Knet, BenchmarkTools
using MNIST: weights, minibatch, xtrn, ytrn
using Knet: gpusync, reluback, xentloss, xentback
using AutoGrad: unbroadcast

atype  = KnetArray{Float32}
hidden = [64]
batch  = 100
data   = minibatch(xtrn, ytrn, batch, atype=atype)
model  = weights(64, atype=atype)

forw = quote
    a1 = w[1]*x                         # 01 0.56 0.56
    a2 = w[2].+a1                       # 02 0.60 0.04
    a3 = relu(a2)                       # 03 0.63 0.03  0.64 0.04
    #a3 = max(0,a2)   
    a4 = w[3]*a3                        # 04 0.76 0.13  0.77 0.13
    a5 = w[4].+a4                       # 05 0.79 0.03  0.80 0.04
    #a6 = y - a5      
    a6 = logp(a5,1)                     # 06 1.02 0.23  1.03 0.23
    #a6 = a5 .- log(sum(exp(a5),1))
    #a7 = a6.*a6
    #a7 = a6.^2
    #a7 = abs2(a6)
    a7 = y .* a6                        # 07 1.06 0.04  1.06 0.03
    #a7 = sumabs2(a6)
    a8 = sum(a7)                        # 08 1.32 0.26  1.35 0.28
    #a6 = xentloss(a5,y,1)              # xx 1.32 0.52
end                                     # 20 1.43 0.11  1.42 0.10 (record)

back = quote
    d7 = ones(a7)                       # 09 1.35 0.03  1.39 0.07
    d6 = unbroadcast(a6, y .* d7)       # 10 1.39 0.04  1.40 0.01
    d5 = d6 - exp(a6) .* sum(d6,1)	# 11 1.57 0.18  1.57 0.17
#    d5 = xentback(a5,y,1)	        # xx 1.63 0.31
    dw4 = unbroadcast(w[4], d5)         # 12 1.68 0.11  1.63 0.06
    d4 = unbroadcast(a4, d5)            # 13 1.69 0.01  1.63 0.00
    dw3 = A_mul_Bc(d4, a3)              # 14 1.85 0.16  1.79 0.16
    d3 = Ac_mul_B(w[3], d4)             # 15 1.96 0.11  1.89 0.10  1.90 0.11
    d2 = reluback(d3,a3)                # 16 1.99 0.03  1.99 0.10  1.93 0.03
    dw2 = unbroadcast(w[2], d2)         # 17 2.04 0.05  2.05 0.06  1.98 0.05
    d1 = unbroadcast(a1, d2)            # 18 2.04 0.00  2.05 0.00  1.99 0.01
    dw1 = A_mul_Bc(d1, x)               # 19 2.22 0.18  2.23 0.18  2.17 0.18
end                                     # 21 2.48 0.26  2.52 0.29  2.52 0.35 (grad)

forw = filter(p->isa(p,Expr), forw.args)
back = filter(p->isa(p,Expr), back.args)
fun = Any[]

for i=1:length(forw)
    f = Symbol("f$i")
    eval(Expr(:function, :($f(w,x,y)), Expr(:block,forw[1:i]...)))
    push!(fun, eval(f))
end
fforw = fun[end]
for i=1:length(back)
    j=i+length(forw)
    f = Symbol("f$j")
    eval(Expr(:function, :($f(w,x,y)), Expr(:block,forw...,back[1:i]...)))
    push!(fun, eval(f))
end

push!(fun,(w,x,y)->AutoGrad.forward_pass(fforw,(w,x,y),(),1))
push!(fun,grad(fforw))

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
