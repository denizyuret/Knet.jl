#CFLAGS=-g -L.
CFLAGS=-O2

libknet.so: drop.o actf.o loss.o update.o kperceptron.o util/linalg.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -lcurand -o $@

%.o: %.cu knet.h
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	-rm *.o libknet.so
