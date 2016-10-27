using Knet;

"""
x (2,2)
0   10
20  30

w (3,3)
1   2   3
4   5   6
7   8   9

y (4,4)
0   10  20  30
20  110 170 150
80  290 350 270
140 370 420 270

How is deconv4 calculated ?

Flipped w (3,3)
9   8   7
6   5   4
3   2   1

Padded input x by windowSize-1 (6,6)
0   0   0   0   0   0
0   0   0   0   0   0
0   0   0   10  0   0
0   0   20  30  0   0
0   0   0   0   0   0
0   0   0   0   0   0

Now apply convolution - there is a better way but this is the easiest...(switch forward and backward passes)

Output y is of size (4,4)
0   10  20  30
20  110 170 150
80  290 350 270
140 370 420 270
"""
y = KnetArray(reshape(Float32[0 10 20 30; 20 110 170 150; 80 290 350 270; 140 370 420 270]))
x = KnetArray(reshape(Float32[0.0 10.0; 20.0 30.0],2,2,1,1))
w = KnetArray(reshape(Float32[1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0],3,3,1,1))
display(x)
display(w)
ycalc = deconv4(w,x)
display(ycalc)

if ycalc == y
    info("deconv4 test passed")
else
    warn("deconv4 test failed!")
    println("actual result: $(y) calculated result: $(ycalc)")
end