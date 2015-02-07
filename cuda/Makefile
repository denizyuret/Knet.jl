CFLAGS=-g
#CC=nvcc
#OBJ=back.o forw.o update.o layer.o forward.o forwback.o
#SRC=back.cu forw.cu update.cu layer.cu forward.cu forwback.cu

test1: test.o libjnet.so 
	nvcc $(CFLAGS) $^ -L. -ljnet -o $@

test2: test.c jnet.cu jnet.h
	nvcc $(CFLAGS) test.c jnet.cu -lcublas -o test2 

libjnet.so: jnet.cu jnet.h
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $< -lcublas -o $@

%.o: %.cu
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -o $@

clean:
	-rm *.o *.so
