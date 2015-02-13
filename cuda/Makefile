CFLAGS=-g -L.

all: libjnet.so libjnet_h5.so

libjnet.so: jnet.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -o $@

libjnet_h5.so: jnet_h5.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lhdf5 -lhdf5_hl -o $@

jnet.o: jnet.cu jnet.h

jnet_h5.o: jnet_h5.cu jnet_h5.h jnet.h

%.o: %.cu
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	-rm *.o *.so
