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

x = reshape([1.0:16.0...],(4,4,1,1));
w = ones(3,3,1,1);
y = reshape([54.0 90.0; 63.0 99.0], (2,2,1,1));


outputm = deconv4(KnetArray(w),KnetArray(y), KnetArray(ones(4,4,1,1))); #Output should be x=(4x4)

display(outputm);


#Normal Convolution
#w = reshape([1.0,2.0,3.0],(3,1,1,1));
#x = reshape([1.0,2.0,3.0,4.0,5.0,6.0,7.0],(7,1,1,1));
#forw = reshape([10.0,16.0,22.0,28.0,34.0],(5,1,1,1));
#back = reshape([3.0,5.0,6.0,6.0,6.0,3.0,1.0],(7,1,1,1));