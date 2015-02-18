CFLAGS=-g -L.

all: libkunet.so libkunet_h5.so

libkunet.so: kunet.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -lcurand -o $@

libkunet_h5.so: kunet_h5.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lhdf5 -lhdf5_hl -o $@

kunet.o: kunet.cu kunet.h

kunet_h5.o: kunet_h5.cu kunet_h5.h kunet.h

%.o: %.cu
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	-rm *.o *.so
