using KUnet

"""
This is the IRNN model, a recurrent net with relu activations whose
recurrent weights are initialized with the identity matrix.  From: Le,
Q. V., Jaitly, N., & Hinton, G. E. (2015). A Simple Way to Initialize
Recurrent Networks of Rectified Linear Units. arXiv preprint
arXiv:1504.00941.
"""
IRNN(n; std=0.01)=Net(Mmul(n; init=randn!, initp=(0,std)), 
                      (Mmul(n; init=eye!), 5), 
                      Add2(), Bias(), Relu())

eye!(a)=copy!(a, eye(eltype(a), size(a)...)) # TODO: don't alloc
