CFLAGS=-g -L../cuda -I../cuda
JNET_H=../cuda/jnet.h ../cuda/jnet_h5.h
JNET_L=../cuda/libjnet.so ../cuda/libjnet_h5.so
MATLAB=matlab -nosplash -nodesktop -r

all: test_y # test_backprop test_train test_adagrad test_dropout...

test_y: test_matlab_y.h5 test_cuda_y.h5 test_caffe_y.h5
	matlab.sh "maxdiff('test_caffe_y.h5', 'test_matlab_y.h5')"
	matlab.sh "maxdiff('test_caffe_y.h5', 'test_cuda_y.h5')"

test_matlab_y.h5: predict.m devx.h5 dev1.h5 dev2.h5
	matlab.sh "predict('devx.h5', 'dev1.h5', 'dev2.h5', 'test_matlab_y.h5')"

test_cuda_y.h5: predict devx.h5 dev1.h5 dev2.h5
	predict -b128 devx.h5 dev1.h5 dev2.h5 test_cuda_y.h5

dev.caffemodel: import_weights.py dev.net dev1.h5 dev2.h5
	python $^ $@

test_caffe_y.h5: predict.py dev.net dev.caffemodel
	-rm $@
	python $^ 82 $@

train: train.o $(JNET_L)
	nvcc $(CFLAGS) $< -ljnet -ljnet_h5 -o $@

backprop: backprop.o $(JNET_L)
	nvcc $(CFLAGS) $< -ljnet -ljnet_h5 -o $@

predict: predict.o $(JNET_L)
	nvcc $(CFLAGS) $< -ljnet -ljnet_h5 -o $@

predict.o: predict.cu $(JNET_H)

backprop.o: backprop.cu $(JNET_H)

train.o: train.cu $(JNET_H)

%.o: %.cu
	nvcc -c $(CFLAGS) $< -o $@

clean:
	-rm *.o test_*

testcuda: testcuda.c ../cuda/jnet.h ../cuda/jnet.cu
	cd ../cuda; make
	gcc -g $< -L /usr/local/cuda/lib64/stubs -L ~/.local/lib64 -L../cuda -I../cuda -I/usr/local/cuda/include -lhdf5_hl -lhdf5 -ljnet -lcuda -o $@

h5getinfo: h5getinfo.c
	gcc -g -std=c99 -I ~/.local/include -L ~/.local/lib64 $< -lhdf5 -lhdf5_hl -o $@

h5diff: h5diff.c
	gcc -g -std=c99 -I ~/.local/include -L ~/.local/lib64 $< -lhdf5 -lhdf5_hl -o $@

