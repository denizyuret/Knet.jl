using Knet, AutoGrad
using Knet: @cuda, cudnnhandle, Cptr, TD, cudnnSoftmaxForward, cudnnSoftmaxBackward, _logp
using BenchmarkTools

# algo=2 corresponds to logp(x,1)
function logp_(x,d...)
    if isa(x,Number)
        return zero(x)
    elseif isempty(x)
        return x
    elseif isempty(d)
        x1 = reshape(x, (1,1,length(x),1))
        y1 = cudnnSoftmaxForward(x1,algo=2)
        reshape(y1,size(x))
    elseif d == (1,)
        cudnnSoftmaxForward(x,algo=2)
    elseif d == (2,)
        x1 = transpose(x)
        y1 = cudnnSoftmaxForward(x1,algo=2)
        transpose(y1)
    else
        error("logp $d not implemented yet.")
    end
end

x1 = ka(randn(Float32,10,100))
x2 = ka(randn(Float32,100,100))
x3 = ka(randn(Float32,1000,100))
x4 = ka(randn(Float32,10000,100))
x5 = ka(randn(Float32,100000,100))

cds()=@cuda(cudart, cudaDeviceSynchronize, ())
rep(x)=clamp(div(10^8,length(x)),10,1000)
logp0(x,d...)=(for i=1:rep(x); _logp(x,d...); end; cds()) # old implementation
logp1(x,d...)=(for i=1:rep(x); logp_(x,d...); end; cds()) # test implementation
logp2(x,d...)=(for i=1:rep(x); logp(x,d...); end; cds())  # new (hybrid) implementation, should be best of the other two
sump0(x,d...)=sum(_logp(x,d...)[1,:]); grad0 = grad(sump0)
sump1(x,d...)=sum(logp_(x,d...)[1,:]); grad1 = grad(sump1)
sump2(x,d...)=sum(logp(x,d...)[1,:]);  grad2 = grad(sump2)
back0(x,d...)=(for i=1:rep(x); grad0(x,d...); end; cds())
back1(x,d...)=(for i=1:rep(x); grad1(x,d...); end; cds())
back2(x,d...)=(for i=1:rep(x); grad2(x,d...); end; cds())
# macro bm(ex); :(println($(sprint(Base.show_unquoted,ex)));show(@benchmark $ex)) end

@show logp(x1) ≈ logp_(x1) ≈ _logp(x1)
@show logp(x1,1) ≈ logp_(x1,1) ≈ _logp(x1,1)
@show logp(x1',2) ≈ logp_(x1',2) ≈ _logp(x1',2)

@show grad0(x1) ≈ grad1(x1) ≈ grad2(x1)
@show grad0(x1,1) ≈ grad1(x1,1) ≈ grad2(x1,1)
@show grad0(x1',2) ≈ grad1(x1',2) ≈ grad2(x1',2)

# dimensions: 0/1/2, forw/back, model0/1, x1:5

for d in (0,1,2)
    for f in (logp0, logp1, logp2, back0, back1, back2)
        @printf("%s(x,%d)",f,d)
        for x in (x1,x2,x3,x4,x5)
            xt = x.'
            knetgc()
            b = (d==0 ? (@benchmark $f($x)) :
                 d==1 ? (@benchmark $f($x,1)) :
                 d==2 ? (@benchmark $f($xt,2)) :
                 error())
            times = b.times ./ rep(x)
            # @printf("\t%dx%d/%.1e/%.1e/%.1e", length(times), rep(x), minimum(times), median(times), mean(times))
            @printf("\t%.1e", minimum(times))
        end
        println()
    end
end

#=
logp0(x,0)	8.5e+04	8.6e+04	1.0e+05	2.8e+05	2.2e+06
logp1(x,0)	1.2e+04	4.7e+04	3.8e+05	5.4e+06	5.3e+07
logp2(x,0)	1.2e+04	4.7e+04	1.0e+05	2.8e+05	2.2e+06
back0(x,0)	2.2e+05	2.5e+05	2.8e+05	6.2e+05	4.5e+06
back1(x,0)	1.5e+05	1.7e+05	7.5e+05	9.5e+06	9.4e+07
back2(x,0)	1.4e+05	1.7e+05	2.3e+05	6.2e+05	4.5e+06
logp0(x,1)	3.3e+04	3.6e+04	5.5e+04	2.8e+05	2.4e+06
logp1(x,1)	1.2e+04	1.3e+04	1.9e+04	1.4e+05	1.2e+06
logp2(x,1)	1.2e+04	1.3e+04	1.9e+04	1.4e+05	1.2e+06
back0(x,1)	1.6e+05	1.7e+05	1.8e+05	6.1e+05	4.8e+06
back1(x,1)	1.3e+05	1.3e+05	1.3e+05	3.5e+05	2.5e+06
back2(x,1)	1.3e+05	1.3e+05	1.3e+05	3.5e+05	2.5e+06
logp0(x,2)	3.3e+04	3.8e+04	9.1e+04	6.3e+05	7.9e+06
logp1(x,2)	2.7e+04	3.0e+04	4.1e+04	2.5e+05	2.1e+06
logp2(x,2)	2.7e+04	3.0e+04	4.1e+04	2.5e+05	2.1e+06
back0(x,2)	1.7e+05	1.7e+05	2.4e+05	1.2e+06	1.3e+07
back1(x,2)	1.7e+05	1.7e+05	1.7e+05	5.8e+05	4.5e+06
back2(x,2)	1.7e+05	1.7e+05	1.7e+05	5.8e+05	4.5e+06
=#

function timesoftmax(x,dir,algo,mode)
    if dir == 0
        (for i=1:rep(x); cudnnSoftmaxForward(x,algo=algo,mode=mode); end; cds())
    else # dir == 1
        (for i=1:rep(x); cudnnSoftmaxBackward(x,x,algo=algo,mode=mode); end; cds())
    end
end

# test other configurations
#=
for dir in (0,1)
    for algo in (0,1,2)
        for mode in (0,1)
            @printf("%s%d%d",dir,algo,mode)
            for x in (x1,x2,x3,x4,x5)
                knetgc()
                b = (@benchmark $timesoftmax($x,$dir,$algo,$mode))
                times = b.times ./ rep(x)
                @printf("\t%.1e", minimum(times))
            end
            println()
        end
    end
end

forw:
000	9.4e+03	1.0e+04	1.6e+04	1.2e+05	9.7e+05 # fast mode has slight advantage in forw
001	9.4e+03	1.0e+04	1.6e+04	1.2e+05	9.7e+05
010	1.2e+04	1.3e+04	2.0e+04	1.4e+05	1.2e+06
011	1.2e+04	1.3e+04	2.0e+04	1.4e+05	1.2e+06
020	1.2e+04	1.3e+04	1.9e+04	1.4e+05	1.2e+06 # log mode is the same as slow mode
021	1.2e+04	1.3e+04	1.9e+04	1.4e+05	1.2e+06
back:
100	9.4e+03	9.3e+03	1.5e+04	9.7e+04	7.4e+05
101	2.0e+05	2.0e+05	4.2e+05	2.6e+06	2.2e+07 # mode=1 has expensive back
110	9.4e+03	9.3e+03	1.5e+04	9.7e+04	7.4e+05
111	2.0e+05	2.0e+05	4.2e+05	2.6e+06	2.2e+07
120	9.4e+03	1.0e+04	1.6e+04	8.7e+04	7.1e+05
121	2.3e+05	2.3e+05	5.6e+05	3.5e+06	2.9e+07
=#
