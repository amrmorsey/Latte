import os
import numpy as np
from PIL import Image

from json_tricks.np import dump

output_path = './image'
image_path = './1.png'

if not os.path.exists(output_path):
    os.makedirs(output_path)

im = Image.open(image_path)
im_data = np.asarray(im)

with open(os.path.join(output_path, 'image.ahsf'), 'w') as mean_file:
    dump(im_data.flatten(), mean_file)

with open(os.path.join(output_path, 'image_shape.ahsf'), 'w') as shape_file:
    shape = im_data[None] if len(im_data.shape) == 2 else im_data.shape
    for shape in im_data[None].T.shape:
        shape_file.write(str(shape) + ' ')

