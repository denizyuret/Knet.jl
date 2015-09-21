CFLAGS=-O3 -I.

libkunet.so: op/drop.o op/actf.o op/loss.o update.o model/kperceptron.o util/linalg.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -lcurand -o $@

%.o: %.cu kunet.h
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	-rm *.o */*.o libkunet.so
