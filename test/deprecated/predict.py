import sys
import h5py
import caffe
import time
import numpy as np

# Usage: python predict.py foo.net foo.caffemodel iters y.h5
# Don't know how to feed in x here as it is hardcoded in the hdf5 layer

caffe.set_phase_test()
caffe.set_mode_gpu()
net = caffe.Net(sys.argv[1], sys.argv[2])
iters = int(sys.argv[3])
ydims = net.blobs['fc2'].data.shape
batch = ydims[0]
ydims = (iters*ydims[0], ydims[1], ydims[2], ydims[3])
y = np.zeros(ydims, dtype=net.blobs['fc2'].data.dtype)
yptr = 0
t = time.time()

for i in range(0,iters):
    net.forward()
    y[yptr:yptr+batch,:,:,:] = net.blobs['fc2'].data
    yptr = yptr+batch

print 'Elapsed: %s' % (time.time() - t)

f = h5py.File(sys.argv[4])
f.create_dataset('data', data=y.squeeze())
f.close()
