test1: test.c libjnet.so jnet.h
	gcc -std=c99 test.c -L. -ljnet -o test1

test2: test.c jnet.cu jnet.h
	nvcc test.c jnet.cu -lcublas -o test2 

libjnet.so: jnet.cu jnet.h
	nvcc --shared --compiler-options -fPIC jnet.cu -lcublas -o libjnet.so
