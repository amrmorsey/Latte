# Adapted from https://stackoverflow.com/a/44061386
import argparse

from caffe.proto import caffe_pb2
from google.protobuf import text_format

parser = argparse.ArgumentParser(
    'Parse original prototxt file into a simplified version. WARNING: caffe\'s upgrade_net_proto_text must be run before inputing the model prototxt to this script.')
parser.add_argument(metavar='model.prototxt', dest='model_prototxt',
                    help='Path to model.prototxt')

args = parser.parse_args()

new_format_model_def = args.model_prototxt

file_name = new_format_model_def.split('.')[0]

parsible_net = caffe_pb2.NetParameter()
text_format.Merge(open(new_format_model_def).read(), parsible_net)

pool_type_dict = ['MAX', 'AVE']

with open('simple_' + file_name + ".ahsf", 'w') as outfile:
    for layer in parsible_net.layer:
        outfile.write(layer.type + " ")
        outfile.write(layer.name + " ")

        if layer.type == 'Input':
            for i in layer.input_param.shape[0].dim:
                outfile.write(str(i) + " ")
        elif layer.type == 'Convolution':
            num_output = layer.convolution_param.num_output
            kernel = layer.convolution_param.kernel_size[0] if len(layer.convolution_param.kernel_size) else 1
            stride = layer.convolution_param.stride[0] if len(layer.convolution_param.stride) else 1
            pad = layer.convolution_param.pad[0] if len(layer.convolution_param.pad) else 0
            outfile.write("{} {} {} {}".format(num_output, kernel, stride, pad))
        elif layer.type == 'Pooling':
            pool = pool_type_dict[layer.pooling_param.pool]
            kernel = layer.pooling_param.kernel_size
            stride = layer.pooling_param.stride
            pad = layer.pooling_param.pad
            outfile.write("{} {} {} {}".format(pool, kernel, stride, pad))
        elif layer.type == "InnerProduct":
            num_output = layer.inner_product_param.num_output
            outfile.write(str(num_output) + " ")

        outfile.write('\n')
