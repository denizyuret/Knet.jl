import sys
import h5py
import caffe
import numpy as np

# Usage: python backprop.py foo.net foo.caffemodel out
# Should perform one batch of backprop
# NOTE: caffe uses 0-based indexing and in particular 0-based labels!

caffe.set_phase_train()
caffe.set_mode_gpu()
net = caffe.Net(sys.argv[1], sys.argv[2])
net.forward()
net.backward()

out=sys.argv[3]
nout = 1
for k,v in net.params.iteritems():
    fname = out + str(nout) + '.h5'
    f = h5py.File(fname, 'w')
    f.create_dataset('w', data=v[0].data.squeeze().transpose())
    f.create_dataset('b', data=v[1].data.squeeze(axis=(1,2)))
    f.create_dataset('dw', data=v[0].diff.squeeze().transpose())
    f.create_dataset('db', data=v[1].diff.squeeze(axis=(1,2)))
    if nout==1:
        f.attrs['f'] = np.string_('relu')
    f.close()
    nout += 1
