CFLAGS=-g -L../cuda -I../cuda
JNET_H=../cuda/jnet.h ../cuda/jnet_h5.h
JNET_L=../cuda/libjnet.so ../cuda/libjnet_h5.so
MATLAB=matlab -nosplash -nodesktop -r
# BP=937
BP=1

all: test_bp # test_y test_train test_adagrad test_dropout...

test_bp: test_bp_matlab1.h5 test_bp_cuda1.h5 test_bp_caffe1.h5
	h5diff test_bp_caffe1.h5 test_bp_cuda1.h5
	h5diff test_bp_caffe2.h5 test_bp_cuda2.h5
	h5diff test_bp_caffe1.h5 test_bp_matlab1.h5
	h5diff test_bp_caffe2.h5 test_bp_matlab2.h5
	h5diff test_bp_cuda1.h5 test_bp_matlab1.h5
	h5diff test_bp_cuda2.h5 test_bp_matlab2.h5

test_bp_matlab1.h5: backprop.m devx.h5 dev1.h5 dev2.h5 devy.h5
	matlab.sh "backprop('x', 'devx.h5', 'y', 'devy.h5', 'net', 'dev1.h5,dev2.h5', 'batch', ${BP}, 'out', 'test_bp_matlab')"

test_bp_cuda1.h5: backprop devx.h5 dev1.h5 dev2.h5 devy.h5
	backprop -b${BP} -o test_bp_cuda devx.h5 dev1.h5 dev2.h5 devy.h5

test_bp_caffe1.h5: backprop.py dev001.net dev.caffemodel
	python $^ test_bp_caffe 2> test_bp_caffe.log

test_y: test_y_matlab.h5 test_y_cuda.h5 test_y_caffe.h5
	h5diff test_y_caffe.h5 test_y_matlab.h5
	h5diff test_y_caffe.h5 test_y_cuda.h5

test_y_matlab.h5: predict.m devx.h5 dev1.h5 dev2.h5
	matlab.sh "predict('devx.h5', 'dev1.h5', 'dev2.h5', 'test_y_matlab.h5')"

test_y_cuda.h5: predict devx.h5 dev1.h5 dev2.h5
	predict -b128 devx.h5 dev1.h5 dev2.h5 test_y_cuda.h5

test_y_caffe.h5: predict.py dev.net dev.caffemodel
	-rm $@
	python $^ 82 $@ 2> test_y_caffe.log

dev.caffemodel: import_weights.py dev.net dev1.h5 dev2.h5
	python $^ $@

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

