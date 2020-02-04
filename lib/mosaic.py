import numpy as np
import numba as nb
from numba import jitclass
from numba import uint8

@nb.njit
def Clac(tmpImage, outputImage, outputWidth, outputHeight):
    for i in range(outputHeight):
        for j in range(outputWidth):
            outputImage[i, j] = tmpImage[i, j, 2] if i % 2 == 0 and j % 2 == 0 \
                           else tmpImage[i, j, 0] if i % 2 != 0 and j % 2 != 0 \
                           else tmpImage[i, j, 1]            
    return outputImage

class Mosaic:
    def __init__(self, oriImage, width, height):
        self.outputWidth = width
        self.outputHeight = height
        self.tmpImage = np.copy(oriImage)
        self.outputImage = np.zeros((self.outputHeight, self.outputWidth), dtype=np.uint8)

    def Algorithm(self):
        return Clac(self.tmpImage, self.outputImage, self.outputWidth, self.outputHeight)
        