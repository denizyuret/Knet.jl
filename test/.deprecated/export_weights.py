# Usage: python export_weights.py foo.net foo.caffemodel out
# Will create layer files out1.h5 out2.h5 ...

import sys
import h5py
import caffe
import numpy as np

net=caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
out=sys.argv[3]
nout=1
for k,v in net.params.iteritems():
    fname = out + str(nout) + '.h5'
    f = h5py.File(fname, 'w')
#    f.create_dataset('w', data=v[0].data.squeeze().transpose())
#    f.create_dataset('b', data=v[1].data.squeeze(axis=(1,2)))
    f.create_dataset('w', data=v[0].data)
    f.create_dataset('b', data=v[1].data)
    if np.any(v[0].diff != 0):
#        f.create_dataset('dw', data=v[0].diff.squeeze().transpose())
#        f.create_dataset('db', data=v[1].diff.squeeze(axis=(1,2)))
        f.create_dataset('dw', data=v[0].diff)
        f.create_dataset('db', data=v[1].diff)
    if nout == 1:
        f.attrs['f'] = np.string_('relu') # for julia
        f.attrs['xfunc'] = np.int32(0)     # for cuda
        f.attrs['yfunc'] = np.int32(1)
    else:
        f.attrs['xfunc'] = np.int32(0)
        f.attrs['yfunc'] = np.int32(2)
    f.close()
    nout += 1
