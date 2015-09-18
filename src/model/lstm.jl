using KUnet

function LSTM(n; fbias=0, dropout=0)
    Net(
        Drop(dropout),	# 1. drop

        (Mmul(n),1),
        (Mmul(n),26),
        Add2(),
        Bias(),
        Sigm(),         # 6. input

        (Mmul(n),1), 
        (Mmul(n),26),
        Add2(),
        Bias(init=fill!, initp=fbias),
        Sigm(),         # 11. forget

        (Mmul(n),1), 
        (Mmul(n),26),
        Add2(),
        Bias(),
        Sigm(),         # 16. output

        (Mmul(n),1), 
        (Mmul(n),26),
        Add2(),
        Bias(),
        Tanh(),         # 21. g
        
        (Mul2(),6,21),
        (Mul2(),11,24),
        Add2(),         # 24. c = i*g + f*c'
    
        Tanh(),
        (Mul2(),16,25)  # 26. h = o*tanh(c)
    )
end

# More compact definition without dropout:
function LSTM1(n; fbias=0)
    m = Net((add2(n),0,13), Sigm(),     	# 1-2. input  (op 1-5)
            (add2(n),0,13), Sigm(),             # 3-4. forget (op 6-10)
            (add2(n),0,13), Sigm(),             # 5-6. output (op 11-15)
            (add2(n),0,13), Tanh(),             # 7-8. g      (op 16-20)
            (Mul2(),2,8),(Mul2(),4,11),Add2(),  # 9-11. c = i*g + f*c[t-1] (op 21-23)
            Tanh(), (Mul2(),6,12))              # 12-13. h = o * tanh(c) (op 24-25)
    fbias != 0 && setparam!(m.op[9]; init=fill!, initp=fbias)
    return m
end

add2(n)=Net(Mmul(n), (Mmul(n),-1), Add2(), Bias())
