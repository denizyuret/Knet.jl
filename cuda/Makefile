CFLAGS=-g -L. -L$(HOME)/.local/lib64
#CC=nvcc
#OBJ=back.o forw.o update.o layer.o forward.o forwback.o
#SRC=back.cu forw.cu update.cu layer.cu forward.cu forwback.cu

predict: predict.o libjnet.so libjnet_h5.so
	nvcc $(CFLAGS) $< -ljnet -ljnet_h5 -o $@

libjnet.so: jnet.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -o $@

libjnet_h5.so: jnet_h5.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lhdf5 -lhdf5_hl -o $@

%.o: %.cu
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -o $@

# test1: test.o libjnet.so 
# 	nvcc $(CFLAGS) $^ -L. -ljnet -o $@

# test2: test.c jnet.cu jnet.h
# 	nvcc $(CFLAGS) test.c jnet.cu -lcublas -o test2 

clean:
	-rm *.o *.so predict
