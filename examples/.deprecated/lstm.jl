lstm(; n=0, fbias=0) = 
quote
    x  = input()
    i  = add2(x,h;n=$n,f=sigm)
    f  = add2(x,h;n=$n,f=sigm,b=$fbias)
    o  = add2(x,h;n=$n,f=sigm)
    g  = add2(x,h;n=$n,f=tanh)
    ig = mul(i,g)
    fc = mul(f,c)
    c  = add(ig,fc)
    tc = tanh(c)
    h  = mul(tc,o)
end

add2(; n=1,f=sigm,b=0) = 
quote
    x1 = input()
    w1 = par($n,0)
    y1 = dot(w1,x1)
    x2 = input()
    w2 = par($n,0)
    y2 = dot(w2,x2)
    x3 = add(y1,y2)
    w3 = par(0; init=Constant($b))
    y3 = add(w3,x3)
    ou = $f(y3)
end


### DEAD CODE

# using Knet

# function LSTM(n; fbias=0, dropout=0)
#     Net(
#         Drop(dropout),	# 1. drop

#         (Mmul(n),1),
#         (Mmul(n),26),
#         Add2(),
#         Bias(),
#         Sigm(),         # 6. input

#         (Mmul(n),1), 
#         (Mmul(n),26),
#         Add2(),
#         Bias(init=fill!, initp=fbias),
#         Sigm(),         # 11. forget

#         (Mmul(n),1), 
#         (Mmul(n),26),
#         Add2(),
#         Bias(),
#         Sigm(),         # 16. output

#         (Mmul(n),1), 
#         (Mmul(n),26),
#         Add2(),
#         Bias(),
#         Tanh(),         # 21. g
        
#         (Mul2(),6,21),
#         (Mul2(),11,24),
#         Add2(),         # 24. c = i*g + f*c'
    
#         Tanh(),
#         (Mul2(),16,25)  # 26. h = o*tanh(c)
#     )
# end

# # More compact definition without dropout:
# function LSTM1(n; fbias=0)
#     m = Net((add2(n),0,13), Sigm(),     	# 1-2. input  (op 1-5)
#             (add2(n),0,13), Sigm(),             # 3-4. forget (op 6-10)
#             (add2(n),0,13), Sigm(),             # 5-6. output (op 11-15)
#             (add2(n),0,13), Tanh(),             # 7-8. g      (op 16-20)
#             (Mul2(),2,8),(Mul2(),4,11),Add2(),  # 9-11. c = i*g + f*c[t-1] (op 21-23)
#             Tanh(), (Mul2(),6,12))              # 12-13. h = o * tanh(c) (op 24-25)
#     fbias != 0 && setparam!(m.op[9]; init=fill!, initp=fbias)
#     return m
# end

# add2(n)=Net(Mmul(n), (Mmul(n),-1), Add2(), Bias())
