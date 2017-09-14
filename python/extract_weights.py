import argparse

import caffe
import numpy as np
from json_tricks.np import dump, dumps, load, loads, strip_comments
import os


def extract_caffe_model(model, weights, output_path):
    """extract caffe model's parameters to numpy array, and write them to files
    Args:
      model: path of '.prototxt'
      weights: path of '.caffemodel'
      output_path: output path of numpy params
    Returns:
      None
    """
    net = caffe.Net(model, weights, caffe.TEST)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, 'shapes.ahsf'), 'w') as shape_file:
        for item in net.params.items():
            name, layer = item
            print('convert layer: ' + name)

            num = 0
            for p in net.params[name]:
                name = name.replace('/', '\\')
                name = name if num == 0 else '{}_bias'.format(name)

                with open(os.path.join(output_path, '{}.ahsf'.format(name)), 'w') as outfile:
                    dump(p.data.flatten(), outfile)

                shape_file.write("{} {} ".format(name, len(p.data.shape)))
                for dim in p.data.T.shape:
                    shape_file.write(str(dim) + " ")
                shape_file.write('\n')
                num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract weights from a .caffemodel and output them in a weights folder')
    parser.add_argument(metavar='model.prototxt', dest='model_prototxt',
                        help='Path to model.prototxt')
    parser.add_argument(metavar='model.caffemodel', dest='model_caffemodel', help='Path to model.caffemodel')

    args = parser.parse_args()
    extract_caffe_model(args.model_prototxt, args.model_caffemodel, "./weights")
