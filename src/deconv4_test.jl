using Knet;

"""
x   4x4
1   2   3   4
5   6   7   8
9   10  11  12
13  14  15  16

w   3x3
1   1   1
1   1   1
1   1   1

y   2x2
54  90
63  99
"""

x = KnetArray(reshape([1.0:16.0...],(4,4,1,1)));
w = KnetArray(ones(3,3,1,1));
y = KnetArray(reshape([54.0 90.0; 63.0 99.0], (2,2,1,1)));


#conv(x,w)
#(4,4)*(3,3)=(4-3+1,4-3+1)=(2,2)

#deconv(y,w)
#(2,2)*(3,3)=(2+3-1,2+3-1)=(4,4)

#=(4,4)
0   0   0   0
0   54  90  0
0   63  99  0
0   0   0   0
=#

#x=10 w=5 y=8
x = KnetArray(ones(10,10,1,1));
w = KnetArray(ones(3,3,1,1));
y = KnetArray(ones(8,8,1,1));