# Usage: python caffemodel2h5.py foo.net foo.caffemodel out.h5
# Will create hdf5 file out.h5 with layers in groups 1,2,...

import sys
import h5py
import caffe
import numpy as np

net=caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)
h5file = h5py.File(sys.argv[3], 'w')
# print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
for k,v in net.params.iteritems():
    f = h5file.create_group(k)
    f.create_dataset('w', data=v[0].data)
    f.create_dataset('b', data=v[1].data)
    if np.any(v[0].diff != 0):
        f.create_dataset('dw', data=v[0].diff)
        f.create_dataset('db', data=v[1].diff)

h5file.close()
