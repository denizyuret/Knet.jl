#CFLAGS=-g -L.
CFLAGS=-O2

libkunet.so: relu.o drop.o bias.o logp.o loss.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -lcurand -o $@

%.o: %.cu kunet.h
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	-rm *.o libkunet.so
