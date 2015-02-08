testcuda: testcuda.c ../cuda/jnet.h ../cuda/jnet.cu
	cd ../cuda; make
	gcc -g $< -L /usr/local/cuda/lib64/stubs -L ~/.local/lib64 -L../cuda -I../cuda -I/usr/local/cuda/include -lhdf5_hl -lhdf5 -ljnet -lcuda -o $@
