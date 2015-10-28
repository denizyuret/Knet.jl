CFLAGS=-O3 -I.

libknet.so: op/actf.o op/loss.o update.o util/linalg.o util/rgen.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -lcurand -o $@

%.o: %.cu knet.h
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	-rm *.o */*.o libknet.so
