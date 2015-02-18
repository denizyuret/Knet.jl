CFLAGS=-g -L../cuda -I../cuda
JNET_H=../cuda/jnet.h ../cuda/jnet_h5.h
JNET_L=../cuda/libjnet.so ../cuda/libjnet_h5.so
MATLAB=matlab -nosplash -nodesktop -r
BATCH=937
ITERS=82

test: test_fw test_bp test_tr test_lr test_l2 test_l1 test_ad test_mo test_ne test_dr


### COMPILATION

dev.caffemodel: import_weights.py dev.net dev1.h5 dev2.h5
	python $^ $@ 

rnd1.h5 rnd2.h5: export_weights.py dev.net rnd.caffemodel
	python $^ rnd 2> /dev/null

rnd.caffemodel: rnd.solver dev.net
	caffe.bin train -solver rnd.solver 2> /dev/null
	mv rnd_iter_0.caffemodel $@

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
	-mv *~ *.caffemodel *.solverstate *.o test_* unused

testcuda: testcuda.c ../cuda/jnet.h ../cuda/jnet.cu
	cd ../cuda; make
	gcc -g $< -L /usr/local/cuda/lib64/stubs -L ~/.local/lib64 -L../cuda -I../cuda -I/usr/local/cuda/include -lhdf5_hl -lhdf5 -ljnet -lcuda -o $@

h5getinfo: h5getinfo.c
	gcc -g -std=c99 -I ~/.local/include -L ~/.local/lib64 $< -lhdf5 -lhdf5_hl -o $@

h5diff: h5diff.c
	gcc -g -std=c99 -I ~/.local/include -L ~/.local/lib64 $< -lhdf5 -lhdf5_hl -o $@


### PREDICT TEST

FWB=${BATCH}
FWI=${ITERS}

test_fw: test_fw_caffe.h5 test_fw_matlab.h5 test_fw_cuda.h5 test_fw_julia.h5 test_fw_juliacpu.h5
	@echo '==> Predict comparison'
	h5diff test_fw_caffe.h5 test_fw_matlab.h5
	h5diff test_fw_caffe.h5 test_fw_cuda.h5
	h5diff test_fw_caffe.h5 test_fw_julia.h5
	h5diff test_fw_caffe.h5 test_fw_juliacpu.h5

test_fw_matlab.h5: predict.m devx.h5 dev1.h5 dev2.h5
	@echo '==> Predict test: matlab'
	matlab.sh "predict('devx.h5', 'dev1.h5', 'dev2.h5', 'test_fw_matlab.h5')"

test_fw_cuda.h5: predict devx.h5 dev1.h5 dev2.h5
	@echo '==> Predict test: cuda'
	predict -b${FWB} devx.h5 dev1.h5 dev2.h5 $@

test_fw_caffe.h5: predict.py dev.net dev.caffemodel
	@echo '==> Predict test: caffe'
	-rm $@
	python $^ ${FWI} $@ 2> test_fw_caffe.log

test_fw_julia.h5: predict.jl devx.h5 dev1.h5 dev2.h5
	@echo '==> Predict test: julia'
	-rm $@
	julia predict.jl devx.h5 dev1.h5 dev2.h5 $@

test_fw_juliacpu.h5: predictcpu.jl devx.h5 dev1.h5 dev2.h5
	@echo '==> Predict test: juliacpu'
	-rm $@
	julia predictcpu.jl devx.h5 dev1.h5 dev2.h5 $@

### BACKPROP TEST

BPB=${BATCH}
BPI=${ITERS}

test_bp: test_bp_matlab1.h5 test_bp_cuda1.h5 test_bp_caffe1.h5
	@echo '==> Backprop comparison'
	h5diff test_bp_caffe1.h5 test_bp_matlab1.h5
	h5diff test_bp_caffe2.h5 test_bp_matlab2.h5
	h5diff test_bp_matlab1.h5 test_bp_cuda1.h5
	h5diff test_bp_matlab2.h5 test_bp_cuda2.h5
	h5diff test_bp_caffe1.h5 test_bp_cuda1.h5
	h5diff test_bp_caffe2.h5 test_bp_cuda2.h5

test_bp_matlab1.h5: backprop.m devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> Backprop test: matlab'
	matlab.sh "backprop('x', 'devx.h5', 'y', 'devy.h5', 'net', 'rnd1.h5,rnd2.h5', 'batch', ${BPB}, 'out', 'test_bp_matlab')"

test_bp_cuda1.h5: backprop devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> Backprop test: cuda'
	backprop -b${BPB} -o test_bp_cuda devx.h5 rnd1.h5 rnd2.h5 devy.h5

test_bp_caffe1.h5: backprop.py dev.net rnd.caffemodel
	@echo '==> Backprop test: caffe'
	python $^ test_bp_caffe 2> test_bp_caffe.log


### SIMPLE TRAINING TEST

TRB=${BATCH}
TRI=${ITERS}

test_tr: test_tr_matlab1.h5 test_tr_cuda1.h5 test_tr_caffe1.h5
	@echo '==> Train comparison'
	h5diff test_tr_caffe1.h5 test_tr_matlab1.h5
	h5diff test_tr_caffe2.h5 test_tr_matlab2.h5
	h5diff test_tr_cuda1.h5 test_tr_matlab1.h5
	h5diff test_tr_cuda2.h5 test_tr_matlab2.h5
	h5diff test_tr_caffe1.h5 test_tr_cuda1.h5
	h5diff test_tr_caffe2.h5 test_tr_cuda2.h5

test_tr_cuda1.h5: train devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> Train test: cuda'
	train -b${TRB} -i${TRI} -o test_tr_cuda devx.h5 rnd1.h5 rnd2.h5 devy.h5

test_tr_matlab1.h5: backprop.m devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> Train test: matlab'
	matlab.sh "trainh5('x', 'devx.h5', 'y', 'devy.h5', 'net', 'rnd1.h5,rnd2.h5', 'batch', ${TRB}, 'iters', ${TRI}, 'out', 'test_tr_matlab')"

test_tr_caffe1.h5: test_tr_caffe_iter_${TRI}.caffemodel export_weights.py
	python export_weights.py dev.net test_tr_caffe_iter_${TRI}.caffemodel test_tr_caffe 2>> test_tr_caffe.log

test_tr_caffe_iter_${TRI}.caffemodel: dev.solver dev.net rnd.caffemodel
	@echo '==> Train test: caffe'
	caffe.bin train -solver dev.solver -weights rnd.caffemodel 2> test_tr_caffe.log


### LEARNING RATE

LR=0.01
LRB=${BATCH}
LRI=${ITERS}

test_lr: test_lr_caffe1.h5 test_lr_cuda1.h5 test_lr_matlab1.h5
	@echo '==> LR comparison'
	h5diff test_lr_caffe1.h5 test_lr_matlab1.h5
	h5diff test_lr_caffe2.h5 test_lr_matlab2.h5
	h5diff test_lr_cuda1.h5 test_lr_matlab1.h5
	h5diff test_lr_cuda2.h5 test_lr_matlab2.h5
	h5diff test_lr_caffe1.h5 test_lr_cuda1.h5
	h5diff test_lr_caffe2.h5 test_lr_cuda2.h5

test_lr_cuda1.h5: train devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> LR test: cuda'
	train -l${LR} -b${LRB} -i${LRI} -o test_lr_cuda devx.h5 rnd1.h5 rnd2.h5 devy.h5

test_lr_matlab1.h5: backprop.m devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> LR test: matlab'
	matlab.sh "trainh5('learningRate', ${LR}, 'x', 'devx.h5', 'y', 'devy.h5', 'net', 'rnd1.h5,rnd2.h5', 'batch', ${LRB}, 'iters', ${LRI}, 'out', 'test_lr_matlab')"

test_lr_caffe1.h5: export_weights.py dev.net test_lr_caffe_iter_${LRI}.caffemodel
	python $^ test_lr_caffe 2>> test_lr_caffe.log

test_lr_caffe_iter_${LRI}.caffemodel: lr.solver dev.net rnd.caffemodel
	@echo '==> LR test: caffe'
	caffe.bin train -solver lr.solver -weights rnd.caffemodel 2> test_lr_caffe.log


### L2 TRAINING

L2=1.0
L2B=${BATCH}
L2I=${ITERS}

test_l2: test_l2_caffe1.h5 test_l2_cuda1.h5 test_l2_matlab1.h5
	@echo '==> L2 comparison'
	h5diff test_l2_caffe1.h5 test_l2_matlab1.h5
	h5diff test_l2_caffe2.h5 test_l2_matlab2.h5
	h5diff test_l2_cuda1.h5 test_l2_matlab1.h5
	h5diff test_l2_cuda2.h5 test_l2_matlab2.h5
	h5diff test_l2_caffe1.h5 test_l2_cuda1.h5
	h5diff test_l2_caffe2.h5 test_l2_cuda2.h5

test_l2_cuda1.h5: train devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> L2 test: cuda'
	train -2 ${L2} -b${L2B} -i${L2I} -o test_l2_cuda devx.h5 rnd1.h5 rnd2.h5 devy.h5

test_l2_matlab1.h5: backprop.m devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> L2 test: matlab'
	matlab.sh "trainh5('L2', ${L2}, 'x', 'devx.h5', 'y', 'devy.h5', 'net', 'rnd1.h5,rnd2.h5', 'batch', ${L2B}, 'iters', ${L2I}, 'out', 'test_l2_matlab')"

test_l2_caffe1.h5: export_weights.py dev.net test_l2_caffe_iter_${L2I}.caffemodel
	python $^ test_l2_caffe 2>> test_l2_caffe.log

test_l2_caffe_iter_${L2I}.caffemodel: l2.solver dev.net rnd.caffemodel
	@echo '==> L2 test: caffe'
	caffe.bin train -solver l2.solver -weights rnd.caffemodel 2> test_l2_caffe.log


### L1 TRAINING

L1=0.01
L1B=${BATCH}
L1I=${ITERS}

test_l1: test_l1_caffe1.h5 test_l1_cuda1.h5 test_l1_matlab1.h5
	@echo '==> L1 comparison'
	h5diff test_l1_caffe1.h5 test_l1_matlab1.h5
	h5diff test_l1_caffe2.h5 test_l1_matlab2.h5
	h5diff test_l1_cuda1.h5 test_l1_matlab1.h5
	h5diff test_l1_cuda2.h5 test_l1_matlab2.h5
	h5diff test_l1_caffe1.h5 test_l1_cuda1.h5
	h5diff test_l1_caffe2.h5 test_l1_cuda2.h5

test_l1_cuda1.h5: train devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> L1 test: cuda'
	train -1 ${L1} -b${L1B} -i${L1I} -o test_l1_cuda devx.h5 rnd1.h5 rnd2.h5 devy.h5

test_l1_matlab1.h5: backprop.m devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> L1 test: matlab'
	matlab.sh "trainh5('L1', ${L1}, 'x', 'devx.h5', 'y', 'devy.h5', 'net', 'rnd1.h5,rnd2.h5', 'batch', ${L1B}, 'iters', ${L1I}, 'out', 'test_l1_matlab')"

test_l1_caffe1.h5: export_weights.py dev.net test_l1_caffe_iter_${L1I}.caffemodel
	python $^ test_l1_caffe 2>> test_l1_caffe.log

test_l1_caffe_iter_${L1I}.caffemodel: l1.solver dev.net rnd.caffemodel
	@echo '==> L1 test: caffe'
	caffe.bin train -solver l1.solver -weights rnd.caffemodel 2> test_l1_caffe.log


### ADAGRAD TRAINING
# There are more significant differences with adagrad.
# I checked after 1 or 2 iterations.
# The problem is small differences in dw, especially if they change the sign, are blown up by adagrad.
# There were 6 sign changes in the second iteration dw between matlab and cuda.
# Nothing we can do, maybe show divergence not too much starting from same dw2.

ADB=${BATCH}
ADI=2

test_ad: test_ad_matlab1.h5 test_ad_cuda1.h5 test_ad_caffe1.h5
	@echo '==> Adagrad comparison: we have numerical unstability in w1'
	h5diff test_ad_caffe1.h5 test_ad_matlab1.h5
	h5diff test_ad_caffe2.h5 test_ad_matlab2.h5
	h5diff test_ad_cuda1.h5 test_ad_matlab1.h5
	h5diff test_ad_cuda2.h5 test_ad_matlab2.h5
	h5diff test_ad_caffe1.h5 test_ad_cuda1.h5
	h5diff test_ad_caffe2.h5 test_ad_cuda2.h5

test_ad_cuda1.h5: train devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> Adagrad test: cuda'
	train -a -b${ADB} -i${ADI} -o test_ad_cuda devx.h5 rnd1.h5 rnd2.h5 devy.h5

test_ad_matlab1.h5: backprop.m devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> Adagrad test: matlab'
	matlab.sh "trainh5('adagrad', 1, 'batch', ${ADB}, 'iters', ${ADI}, 'x', 'devx.h5', 'y', 'devy.h5', 'net', 'rnd1.h5,rnd2.h5', 'out', 'test_ad_matlab')"

test_ad_caffe1.h5: export_weights.py dev.net test_ad_caffe_iter_${ADI}.caffemodel
	python $^ test_ad_caffe 2>> test_ad_caffe.log

test_ad_caffe_iter_${ADI}.caffemodel: ad.solver dev.net rnd.caffemodel
	@echo '==> Adagrad test: caffe'
	caffe.bin train -solver ad.solver -weights rnd.caffemodel 2> test_ad_caffe.log


### MOMENTUM TRAINING

MO=0.95
MOB=${BATCH}
MOI=${ITERS}

test_mo: test_mo_caffe1.h5 test_mo_cuda1.h5 test_mo_matlab1.h5
	@echo '==> MO comparison'
	h5diff test_mo_caffe1.h5 test_mo_matlab1.h5
	h5diff test_mo_caffe2.h5 test_mo_matlab2.h5
	h5diff test_mo_cuda1.h5 test_mo_matlab1.h5
	h5diff test_mo_cuda2.h5 test_mo_matlab2.h5
	h5diff test_mo_caffe1.h5 test_mo_cuda1.h5
	h5diff test_mo_caffe2.h5 test_mo_cuda2.h5

test_mo_cuda1.h5: train devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> MO test: cuda'
	train -m ${MO} -b${MOB} -i${MOI} -o test_mo_cuda devx.h5 rnd1.h5 rnd2.h5 devy.h5

test_mo_matlab1.h5: backprop.m devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> MO test: matlab'
	matlab.sh "trainh5('momentum', ${MO}, 'x', 'devx.h5', 'y', 'devy.h5', 'net', 'rnd1.h5,rnd2.h5', 'batch', ${MOB}, 'iters', ${MOI}, 'out', 'test_mo_matlab')"

test_mo_caffe1.h5: export_weights.py dev.net test_mo_caffe_iter_${MOI}.caffemodel
	python $^ test_mo_caffe 2>> test_mo_caffe.log

test_mo_caffe_iter_${MOI}.caffemodel: mo.solver dev.net rnd.caffemodel
	@echo '==> MO test: caffe'
	caffe.bin train -solver mo.solver -weights rnd.caffemodel 2> test_mo_caffe.log


### NESTEROV TRAINING

NE=0.95
NEB=${BATCH}
NEI=${ITERS}

test_ne: test_ne_caffe1.h5 test_ne_cuda1.h5 test_ne_matlab1.h5
	@echo '==> NE comparison'
	h5diff test_ne_caffe1.h5 test_ne_matlab1.h5
	h5diff test_ne_caffe2.h5 test_ne_matlab2.h5
	h5diff test_ne_cuda1.h5 test_ne_matlab1.h5
	h5diff test_ne_cuda2.h5 test_ne_matlab2.h5
	h5diff test_ne_caffe1.h5 test_ne_cuda1.h5
	h5diff test_ne_caffe2.h5 test_ne_cuda2.h5

test_ne_cuda1.h5: train devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> NE test: cuda'
	train -n -m ${NE} -b${NEB} -i${NEI} -o test_ne_cuda devx.h5 rnd1.h5 rnd2.h5 devy.h5

test_ne_matlab1.h5: backprop.m devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> NE test: matlab'
	matlab.sh "trainh5('nesterov', 1, 'momentum', ${NE}, 'x', 'devx.h5', 'y', 'devy.h5', 'net', 'rnd1.h5,rnd2.h5', 'batch', ${NEB}, 'iters', ${NEI}, 'out', 'test_ne_matlab')"

test_ne_caffe1.h5: export_weights.py dev.net test_ne_caffe_iter_${NEI}.caffemodel
	python $^ test_ne_caffe 2>> test_ne_caffe.log

test_ne_caffe_iter_${NEI}.caffemodel: ne.solver dev.net rnd.caffemodel
	@echo '==> NE test: caffe'
	caffe.bin train -solver ne.solver -weights rnd.caffemodel 2> test_ne_caffe.log


### DROPOUT TRAINING

DR=0.75
DRB=${BATCH}
DRI=${ITERS}

test_dr: test_dr_caffe1.h5 test_dr_cuda1.h5 test_dr_matlab1.h5
	@echo '==> DR comparison: note matlab has different rng'
	h5diff test_dr_caffe1.h5 test_dr_matlab1.h5
	h5diff test_dr_caffe2.h5 test_dr_matlab2.h5
	h5diff test_dr_cuda1.h5 test_dr_matlab1.h5
	h5diff test_dr_cuda2.h5 test_dr_matlab2.h5
	h5diff test_dr_caffe1.h5 test_dr_cuda1.h5
	h5diff test_dr_caffe2.h5 test_dr_cuda2.h5

test_dr_cuda1.h5: train devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> DR test: cuda'
	train -d ${DR} -b${DRB} -i${DRI} -o test_dr_cuda devx.h5 rnd1.h5 rnd2.h5 devy.h5

test_dr_matlab1.h5: backprop.m devx.h5 rnd1.h5 rnd2.h5 devy.h5
	@echo '==> DR test: matlab'
	matlab.sh "trainh5('dropout', ${DR}, 'x', 'devx.h5', 'y', 'devy.h5', 'net', 'rnd1.h5,rnd2.h5', 'batch', ${DRB}, 'iters', ${DRI}, 'out', 'test_dr_matlab')"

test_dr_caffe1.h5: export_weights.py dr.net test_dr_caffe_iter_${DRI}.caffemodel
	python $^ test_dr_caffe 2>> test_dr_caffe.log

test_dr_caffe_iter_${DRI}.caffemodel: dr.solver dr.net rnd.caffemodel
	@echo '==> DR test: caffe'
	caffe.bin train -solver dr.solver -weights rnd.caffemodel 2> test_dr_caffe.log

