using KUnet

IRNN(n)=Net(Mmul(n), (Mmul(n; init=eye!), 5), Add2(), Bias(), Relu())

eye!(a)=copy!(a, eye(eltype(a), size(a)...)) # TODO: don't alloc
