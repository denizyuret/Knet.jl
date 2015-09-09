using Base.Test
using CUDArt
using KUnet
import KUnet: forw, back, ninputs

include("../src/rnn.jl")

irnn(h)=RNN(Mmul(h), (Mmul(h),5), Add2(), Bias(), Relu())
add1(h)=RNN(Mmul(h), (Mmul(h),-1), Add2(), Bias(), Sigm())
add2(h)=RNN(Mmul(h), (Mmul(h),-1), Add2(), Bias(), Tanh())
lstm(h)=RNN((add1(h),0,9),      # 1. i
            (add1(h),0,9),      # 2. f
            (add1(h),0,9),      # 3. o
            (add2(h),0,9),      # 4. cc
            (Mul2(),1,4),       # 5. i*cc
            (Mul2(),2,7),       # 6. f*c[t-1]
            Add2(),             # 7. c
            Tanh(),             # 8. tanh(c)
            (Mul2(),3,8))       # 9. o * tanh(c)

function testops(a,ops)
    nops(a) == length(ops) || return false
    for i=1:nops(a)
        isa(a.op[i], ops[i]) || return false
    end
    return true
end

a = irnn(10)
aops = [Mmul, Mmul, Add2, Bias, Relu]
@test testops(a, aops)
@test a.inputs == Any[[6],[5],[1,2],[3],[4]]
@test a.ninputs == 1
@test find(a.save) == [5,6]
#@test a.y == [1,2,1,1,1,3]
#@test a.dy == [5,4,5,5,6,7]
@test all(a.reg .== nothing)
@test a.buffer == a.reg
@test a.stack == Any[]
@test a.sp == 0

b = RNN(irnn(10),irnn(10))
bops = vcat(aops,aops)
@test testops(b, bops)
@test b.inputs == Any[[11],[5],[1,2],[3],[4],[5],[10],[6,7],[8],[9]]
@test b.ninputs == 1
@test find(b.save) == [5,10,11]
#@test b.y == [1,2,1,1,1,3,4,3,3,3,5]
#@test b.dy == [7,6,7,7,8,10,9,10,10,11,12]
@test all(b.reg .== nothing)
@test b.buffer == b.reg
@test b.stack == Any[]
@test b.sp == 0

c = lstm(10)
cops = [Mmul,Mmul,Add2,Bias,Sigm,Mmul,Mmul,Add2,Bias,Sigm,Mmul,Mmul,Add2,Bias,Sigm,Mmul,Mmul,Add2,Bias,Tanh,Mul2,Mul2,Add2,Tanh,Mul2]
@test testops(c, cops)
@test c.inputs == Any[[26],[25],[1,2],[3],[4],[26],[25],[6,7],[8],[9],[26],[25],[11,12],[13],[14],[26],[25],[16,17],[18],[19],[5,20],[10,23],[21,22],[23],[15,24]]
@test c.ninputs == 1
@test find(c.save) == [5,10,15,20,23,24,25,26]
#@test c.y == [1,2,1,1,1,3,4,3,3,3,5,6,5,5,5,7,8,7,7,7,9,10,9,11,12,13]
#@test c.dy == [15,14,15,15,15,17,16,17,17,17,19,18,19,19,19,21,20,21,21,21,22,23,24,25,26,27]
@test all(c.reg .== nothing)
@test c.buffer == c.reg
@test c.stack == Any[]
@test c.sp == 0


# x = KUdense{CudaArray}(rand(3,5))
# y = forw(net, x)
# :ok

            

# mbr(nh)=RNN(Mmul(nh),Bias(),Relu())
# mbr2(n1,n2)=RNN(mbr(n1),mbr(n2))
# foo1(nh,ny)=RNN(Mmul(nh), (Mmul(nh),5), Add2(), Bias(), Relu(), Mmul(ny), Bias())
# foo2(nh)=RNN(Mmul(nh), (Mmul(nh),-1), Add2(), Bias(), Sigm())
# foo3(nh)=RNN(Mmul(nh), (Mmul(nh),-1), Add2(), Bias(), Tanh())
# foo9(nh)=RNN((foo2(nh),0,9), (foo2(nh),0,9), (foo2(nh),0,9), (foo3(nh),0,9),
#              (Mul2(),1,4), (Mul2(),2,7), Add2(), Tanh(), (Mul2(),3,8))

# nh = 3
# ny = 2
# rnn = RNN(Mmul(nh), (Mmul(nh),5), Add2(), Bias(), Relu(), Mmul(ny), Bias(), XentLoss())
# # x = [KUdense{CudaArray}(rand(2)) for t=1:5]
# # y = forw(rnn, x)

# # @show size(rnn.h),length(rnn.h),length(unique(rnn.h))

# lstm = RNN((Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Sigm(), # 1-5 input gate
#            (Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Sigm(), # 6-10 forget gate
#            (Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Sigm(), # 11-15 output gate
#            (Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Tanh(), # 16-20 new memory cell c_tilde
#            (Mul2(),5,20), (Mul2(),10,23), Add2(), Tanh(),       # 21-24 final memory cell c_t
#            (Mul2(),15,24),                                      # 25 output h_t
#            # Mmul(ny), Bias(), XentLoss(),                      # 26-28 loss layers
#            )
          
# # y = forw(lstm, x)

# # @show size(lstm.h),length(lstm.h),length(unique(lstm.h))

# # gru = RNN((Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Sigm(), # 1-5 reset gate r[j,t]
# #           (Mmul(nh),0), (Mmul(nh),25), Add2(), Bias(), Sigm(), # 6-10 update gate z[j,t]
# #           (Mul2(),5,25), Mmul(nh), (Mmul(nh),0), Add2(), Bias(), Tanh(), # 11-16 candidate activation h_tilde[j,t]
# # need to compute 1-z for gru

