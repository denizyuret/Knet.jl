import sys
import h5py
import caffe
import numpy as np

# Usage: python predict.py foo.net foo.caffemodel iters y.h5
# Don't know how to feed in x here as it is hardcoded in the hdf5 layer

caffe.set_phase_test()
caffe.set_mode_gpu()
net = caffe.Net(sys.argv[1], sys.argv[2])
iters = int(sys.argv[3])
y = np.array([])
for i in range(0,iters):
    net.forward()
    y1 = net.blobs['fc2'].data.squeeze().copy()
    if y.size == 0:
        y = y1
    else:
        y = np.concatenate((y,y1))

f = h5py.File(sys.argv[4])
f.create_dataset('data', data=y)
f.close()
