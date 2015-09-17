using KUnet

LSTM(n)=Net((add2(n),0,13), Sigm(),     # 1-2. input
            (add2(n),0,13), Sigm(),     # 3-4. forget
            (add2(n),0,13), Sigm(),     # 5-6. output
            (add2(n),0,13), Tanh(),     # 7-8. cc
            (Mul2(),2,8), (Mul2(),4,11), Add2(), # 9-11. c = i*cc + f*c[t-1]
            Tanh(), (Mul2(),6,12))      # 12-13. h = o * tanh(c)

add2(n)=Net(Mmul(n), (Mmul(n),-1), Add2(), Bias())
