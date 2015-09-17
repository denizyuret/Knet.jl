using KUnet

"""
This is the data generator for the adding problem from: Le, Q. V.,
Jaitly, N., & Hinton, G. E. (2015). A Simple Way to Initialize
Recurrent Networks of Rectified Linear Units. arXiv preprint
arXiv:1504.00941.
"""
type Adding <: Data; len; batchsize; epochsize; rng;
    Adding(len, batchsize, epochsize; rng=MersenneTwister())=new(len, batchsize, epochsize, rng)
end

start(a::Adding)=0

done(a::Adding,n)=(n >= a.epochsize)

function next(a::Adding, n)
    nb = min(a.batchsize, a.epochsize-n)
    x = [ vcat(rand(a.rng,Float32,1,nb),zeros(Float32,1,nb)) for t=1:a.len ]
    y = Array(Float32,1,nb)
    t1 = rand(a.rng,1:a.len,nb)
    t2 = rand(a.rng,1:a.len,nb)
    for b=1:nb
        while t2[b]==t1[b]
            t2[b]=rand(a.rng,1:a.len)
        end
        x[t1[b]][2,b]=1
        x[t2[b]][2,b]=1
        y[b] = x[t1[b]][1,b] + x[t2[b]][1,b]
    end
    return ((x,y), n+nb)
end
