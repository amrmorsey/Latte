import caffe
import numpy as np

caffe.set_mode_cpu()

# Prepare Network
MODEL_FILE = 'test28Gray.2conv.2fc.prototxt'
PRETRAINED = 'test28Gray.2conv.2fc.caffemodel'
MEAN_FILE = 'test28Gray.2conv.2fc.mean.binaryproto'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

blob = caffe.proto.caffe_pb2.BlobProto()
dataBlob = open( MEAN_FILE , 'rb' ).read()
blob.ParseFromString(dataBlob)
dataMeanArray = np.array(caffe.io.blobproto_to_array(blob))

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', dataMeanArray[0].mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255.0)

IMAGE_PATH = '1.png'
input_image = caffe.io.load_image(IMAGE_PATH, color=False)


net.blobs['data'].data[...] = transformer.preprocess('data', input_image)

prediction = net.forward()
pass

# import numpy as np
#
# with open('out.txt', 'w') as f:
#     np.savetxt(f, net.blobs['conv1'].data[0,0], fmt='%.4f', delimiter='\n')