import os
import caffe
import numpy as np

from json_tricks.np import dump


output_path = './mean'

if not os.path.exists(output_path):
    os.makedirs(output_path)

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('test28Gray.2conv.2fc.mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))

# Binaryproto has shape of (1, 1, 28, 28), arr[0] to get (1, 28, 28)
out = arr[0].T

with open(os.path.join(output_path, 'mean2.ahsf'), 'w') as mean_file:
    dump(out.flatten(), mean_file)

with open(os.path.join(output_path, 'mean_shape2.ahsf'), 'w') as shape_file:
    for shape in out.shape:
        shape_file.write(str(shape) + ' ')

