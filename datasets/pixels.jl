using KUnet: Data, nextidx
import Base: start, next, done

"""
This is the data generator for the pixel-by-pixel MNIST problem from:
Le, Q. V., Jaitly, N., & Hinton, G. E. (2015). A Simple Way to
Initialize Recurrent Networks of Rectified Linear Units. arXiv
preprint arXiv:1504.00941.
"""
type Pixels <: Data; x; rng; datasize; epochsize; batchsize; bootstrap; shuffle; batch;
    function Pixels(x...; rng=MersenneTwister(), epoch=ccount(x[1]), batch=16, bootstrap=false, shuffle=false)
        nx = ccount(x[1])
        all(xi->ccount(xi)==nx, x) || error("Item count mismatch")
        idx = (shuffle ? shuffle!(rng,[1:nx;]) : nothing)
        xbatch = [ similar(x[1], (1,batch)) for i=(1:clength(x[1])) ]
        ybatch = similar(x[2], (clength(x[2]),batch))
        new(x, rng, nx, epoch, batch, bootstrap, idx, (xbatch,ybatch))
    end
end

start(d::Pixels)=(d.shuffle != nothing && shuffle!(d.rng, d.shuffle); 0)

done(d::Pixels, n)=(n >= d.epochsize)

function next(d::Pixels, n)
    idx = nextidx(d,n)
    nb = length(idx)
    nt = clength(d.x[1])
    for b=1:nb
        i=idx[b]
        t0 = (i-1)*nt
        @inbounds for t=1:nt
            d.batch[1][t][b] = d.x[1][t0 + t]
        end
        d.batch[2][:,b] = d.x[2][:,i]
    end
    (d.batch, n+nb)
end
