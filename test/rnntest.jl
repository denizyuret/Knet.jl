using CUDArt
using KUnet

nh = 3
ny = 2
rnn = RNN(Mmul(nh), (Mmul(nh),5), Add2(), Bias(), Relu(), Mmul(ny), Bias(), XentLoss())
x = [KUdense{CudaArray}(rand(2)) for t=1:5]
y = forw(rnn, x)

@show size(rnn.h),length(rnn.h),length(unique(rnn.h))

lstm = RNN((Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Sigm(), # 1-5 input gate
           (Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Sigm(), # 6-10 forget gate
           (Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Sigm(), # 11-15 output gate
           (Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Tanh(), # 16-20 new memory cell c_tilde
           (Mul2(),5,20), (Mul2(),10,23), Add2(), Tanh(),       # 21-24 final memory cell c_t
           (Mul2(),15,24), Mmul(ny), Bias(), XentLoss())        # 25-28 output h_t

y = forw(lstm, x)

@show size(lstm.h),length(lstm.h),length(unique(lstm.h))

# gru = RNN((Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Sigm(), # 1-5 reset gate r[j,t]
#           (Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Sigm(), # 6-10 update gate z[j,t]
#           (Mul2(),5,25), Mmul(nh), (Mmul(nh),0), Add2(), Bias(), Tanh(), # 11-16 candidate activation h_tilde[j,t]
# need to compute 1-z for gru

