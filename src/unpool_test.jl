using Knet;

x = zeros(2,2,2,2);
x[:,:,1,1] = [1.0 2.0; 3.0 4.0];
x[:,:,2,1] = [11.0 12.0; 13.0 14.0];

x[:,:,1,2] = [5.0 6.0; 7.0 8.0];
x[:,:,2,2] = [15.0 16.0; 17.0 18.0];

display(x);

y = unpool(x);

display(y);