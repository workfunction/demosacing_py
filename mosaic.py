import numpy as np
import numba as nb

@nb.njit
def Clac(tmpImage, outputImage, outputWidth, outputHeight):
    for i in range(outputHeight):
        for j in range(outputWidth):
            if i % 2 == 0: # even line B, G
                if (j % 2 == 0): # even pixel B
                    outputImage[i, j] = tmpImage[i, j, 2]
                else:            #  odd pixel G
                    outputImage[i, j] = tmpImage[i, j, 1]

            else:         # odd line G, R
                if j % 2 == 0: # even pixel G
                    outputImage[i, j] = tmpImage[i, j, 1]
                else:            #  odd pixel R
                    outputImage[i, j] = tmpImage[i, j, 0]
    return outputImage

class Mosaic:
    def __init__(self, oriImage, width, height):
        self.outputWidth = width
        self.outputHeight = height
        self.tmpImage = np.copy(oriImage)
        self.outputImage = np.zeros((self.outputHeight, self.outputWidth), int)

    def Algorithm(self):
        return Clac(self.tmpImage, self.outputImage, self.outputWidth, self.outputHeight)
        