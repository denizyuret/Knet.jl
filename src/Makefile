#CFLAGS=-g -L.
CFLAGS=-O2

libkunet.so: drop.o actf.o loss.o update.o kperceptron.o util/linalg.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -lcurand -o $@

%.o: %.cu kunet.h
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	-rm *.o libkunet.so
