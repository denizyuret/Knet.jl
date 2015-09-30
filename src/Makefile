CFLAGS=-O3 -I.

libknet.so: op/drop.o op/actf.o op/loss.o update.o model/kperceptron.o util/linalg.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -lcurand -o $@

%.o: %.cu knet.h
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	-rm *.o */*.o libknet.so
