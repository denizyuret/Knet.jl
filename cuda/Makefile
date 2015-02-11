CFLAGS=-g -L. -L$(HOME)/.local/lib64
#CC=nvcc
#OBJ=back.o forw.o update.o layer.o forward.o forwback.o
#SRC=back.cu forw.cu update.cu layer.cu forward.cu forwback.cu

train: train.o libjnet.so libjnet_h5.so
	nvcc $(CFLAGS) $< -ljnet -ljnet_h5 -o $@

backprop: backprop.o libjnet.so libjnet_h5.so
	nvcc $(CFLAGS) $< -ljnet -ljnet_h5 -o $@

predict: predict.o libjnet.so libjnet_h5.so
	nvcc $(CFLAGS) $< -ljnet -ljnet_h5 -o $@

libjnet.so: jnet.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -o $@

libjnet_h5.so: jnet_h5.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lhdf5 -lhdf5_hl -o $@

jnet.o: jnet.cu jnet.h

jnet_h5.o: jnet_h5.cu jnet_h5.h jnet.h

predict.o: predict.cu jnet_h5.h jnet.h

backprop.o: backprop.cu jnet_h5.h jnet.h

train.o: train.cu jnet_h5.h jnet.h

%.o: %.cu
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $< -o $@

clean:
	-rm *.o *.so predict
