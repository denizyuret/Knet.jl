# Usage: python import_weights.py netfile layer1.h5 layer2.h5 ... output
# Will create a file with caffemodel extension.

# Once created the caffemodel can be tested with:
# caffe.bin test -model dev.net -weights dev.caffemodel -gpu 0 -iterations 600

# Using dir() to figure out the interface:
# net: Net
# net.params: OrderedDict
# net.params['fc1']: BlobVec
# net.params['fc1'][0]: Blob
# net.params['fc1'][0].data: array
# net.params['fc1'][0].diff: array
# net.params['fc1'][0].data.shape: (1, 1, 20000, 1326)
# net.params['fc1'][1].data.shape: (1, 1, 1, 20000)

import sys
import h5py
import caffe

net=caffe.Net(sys.argv[1], caffe.TRAIN)
argi=2

for k,v in net.params.iteritems():
    f = h5py.File(sys.argv[argi], 'r')
    v[0].data[...] = f['w'][...].transpose().reshape(v[0].data.shape)
    v[1].data[...] = f['b'][...].reshape(v[1].data.shape)
    f.close()
    argi += 1

net.save(sys.argv[argi])
