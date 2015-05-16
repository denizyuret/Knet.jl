#CFLAGS=-g -L.
CFLAGS=-O2

libkunet.so: drop.o logp.o logploss.o softloss.o xentloss.o util.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -lcublas -lcurand -o $@

%.o: %.cu kunet.h
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

clean:
	-rm *.o libkunet.so
