import numpy as np
from PIL import Image
from scipy.signal import convolve2d

v_filter = [
    [1, 1, 1, 0, -1, -1, -1],
    [1, 1, 1, 0, -1, -1, -1],
    [1, 1, 1, 0, -1, -1, -1],
    [1, 1, 1, 0, -1, -1, -1],
    [1, 1, 1, 0, -1, -1, -1],
    [1, 1, 1, 0, -1, -1, -1],
]

h_filter = [[1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1]]

image = Image.open('/Users/Shafou/Desktop/block.jpg').convert('1')

image_array = np.array(image)

v_conv_image = convolve2d(image_array, v_filter)
h_conv_image = convolve2d(image_array, h_filter)

v_conv_image = np.array(v_conv_image, dtype=np.float32)
h_conv_image = np.array(h_conv_image, dtype=np.float32)

v_conv_image = 255 * (
    v_conv_image - np.min(v_conv_image)) / np.ptp(v_conv_image).astype(int)
h_conv_image = 255 * (
    h_conv_image - np.min(h_conv_image)) / np.ptp(h_conv_image).astype(int)

conv_image = v_conv_image + h_conv_image

image = Image.fromarray(conv_image)
image.show()
