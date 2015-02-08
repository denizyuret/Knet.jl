CFLAGS=-g
#CC=nvcc
#OBJ=back.o forw.o update.o layer.o forward.o forwback.o
#SRC=back.cu forw.cu update.cu layer.cu forward.cu forwback.cu

libjnet.so: jnet.cu jnet.h
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $< -lcublas -o $@

# %.o: %.cu
# 	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -o $@

# test1: test.o libjnet.so 
# 	nvcc $(CFLAGS) $^ -L. -ljnet -o $@

# test2: test.c jnet.cu jnet.h
# 	nvcc $(CFLAGS) test.c jnet.cu -lcublas -o test2 

clean:
	-rm *.o *.so
