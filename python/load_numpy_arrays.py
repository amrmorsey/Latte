import numpy as np
from json_tricks.np import load

image = load('./image/image.ahsf')

arr1 = load('./weights/conv1.ahsf')
arr2 = load('./weights/conv2.ahsf')

out = np.dot(arr1, arr2)
print(out)
