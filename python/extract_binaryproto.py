import argparse
import os
import caffe
import numpy as np

from json_tricks.np import dump

parser = argparse.ArgumentParser('Extract pixel and shape from a mean mean.binaryproto file')
parser.add_argument(metavar='mean_path', dest='mean',
                    help='Path to model.prototxt')

args = parser.parse_args()

output_path = './mean'

if not os.path.exists(output_path):
    os.makedirs(output_path)

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(args.mean, 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))

# Binaryproto has shape of (1, 1, 28, 28), arr[0] to get (1, 28, 28)
out = arr[0]

with open(os.path.join(output_path, 'mean.ahsf'), 'w') as mean_file:
    dump(out.flatten(), mean_file)

with open(os.path.join(output_path, 'mean_shape.ahsf'), 'w') as shape_file:
    for shape in out.T.shape:
        shape_file.write(str(shape) + ' ')

