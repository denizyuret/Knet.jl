type LRN <: Op; n; alpha; beta; k; end
#LRN(x,y;n=5,alpha=1e-4,beta=0.75,k=2.0,o...)=(LRN(n,alpha,beta,k),x,y)
LRN(x,y;o...)=error(:LRN_NOT_TESTED)
ninputs(::LRN)=1
overwrites(::LRN)=false
back_reads_x(::LRN)=true
back_reads_y(::LRN)=true
infersize(::LRN,xdims,ydims)=infersize(Sigm(),xdims,ydims)
forw(l::LRN, x, y; o...)=
    (cudnnLRNCrossChannelForward(x,y; n=l.n, alpha=l.alpha, beta=l.beta, k=l.k); gpusync(); y)
back(l::LRN, dy, dx; x=nothing, y=nothing, o...)=
    (dx!=nothing && cudnnLRNCrossChannelBackward(y,dy,x,dx;n=l.n, alpha=l.alpha, beta=l.beta, k=l.k); gpusync(); dx)
