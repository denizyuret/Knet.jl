add2(n)=RNN(Mmul(n), (Mmul(n),-1), Add2(), Bias())

lstm(n)=RNN((add2(n),0,13), Sigm(),     # 1-2. input
            (add2(n),0,13), Sigm(),     # 3-4. forget
            (add2(n),0,13), Sigm(),     # 5-6. output
            (add2(n),0,13), Tanh(),     # 7-8. cc
            (Mul2(),2,8), (Mul2(),4,11), Add2(), # 9-11. c = i*cc + f*c[t-1]
            Tanh(), (Mul2(),6,12))      # 12-13. h = o * tanh(c)

irnn(n)=RNN(Mmul(n), (Mmul(n; init=initeye), 5), Add2(), Bias(), Relu())

for (layer, op) in 
    ((:sigmlayer, :Sigm),
     (:tanhlayer, :Tanh),
     (:relulayer, :Relu),
     (:softlayer, :Soft),
     (:logplayer, :Logp),
     (:quadlosslayer, :QuadLoss),
     (:softlosslayer, :SoftLoss),
     (:logplosslayer, :LogpLoss),
     (:xentlosslayer, :XentLoss),
     (:perclosslayer, :PercLoss),
     (:scallosslayer, :ScalLoss),
     )
    @eval $layer(n)=RNN(Mmul(n), Bias(), $op())
end
