import numpy as np
import numba as nb
from numba import jitclass
from numba import uint8

spec = [
    ('outputWidth', nb.int64),
    ('outputHeight', nb.int64),
    ('tmpImage', nb.uint8[:, :, :]),
    ('outputImage', nb.uint8[:, :]),
]

@jitclass(spec)
class Mosaic:
    def __init__(self, oriImage, width, height):
        self.outputWidth = width
        self.outputHeight = height
        self.tmpImage = np.copy(oriImage)
        self.outputImage = np.zeros((self.outputHeight, self.outputWidth), dtype=np.uint8)

    def Algorithm(self):
        for i in range(self.outputHeight):
            for j in range(self.outputWidth):
                self.outputImage[i, j] = self.tmpImage[i, j, 2] if i % 2 == 0 and j % 2 == 0 \
                            else self.tmpImage[i, j, 0] if i % 2 != 0 and j % 2 != 0 \
                            else self.tmpImage[i, j, 1]            
        return self.outputImage
        