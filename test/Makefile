testcuda: testcuda.c ../cuda/jnet.h ../cuda/libjnet.so
	gcc -g $< -L ~/.local/lib64 -L../cuda -I../cuda -lhdf5_hl -lhdf5 -ljnet -o $@
