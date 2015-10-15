"""
This is the IRNN model, a recurrent net with relu activations whose
recurrent weights are initialized with the identity matrix.  From: Le,
Q. V., Jaitly, N., & Hinton, G. E. (2015). A Simple Way to Initialize
Recurrent Networks of Rectified Linear Units. arXiv preprint
arXiv:1504.00941.
"""

irnn(;n=1,std=0.01) = quote
    x1 = input()
    w1 = par($n,0; init=Gaussian(0,$std))
    x2 = dot(w1,x1)
    w6 = par($n,0; init=Identity())
    x3 = dot(w6,x6)
    x4 = add(x2,x3)
    w4 = par(0; init=Constant(0))
    x5 = add(w4,x4)
    x6 = relu(x5)
end

# IRNN(n; std=0.01)=Net(Mmul(n; init=randn!, initp=(0,std)), 
#                       (Mmul(n; init=eye!), 5), 
#                       Add2(), Bias(), Relu())

# eye!(a)=copy!(a, eye(eltype(a), size(a)...)) # 
