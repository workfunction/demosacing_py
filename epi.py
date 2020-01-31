import numpy as np
import numba as nb
from numba import jitclass
from numba import int64
from numba import uint8
from numba import float64

spec = [
    ('width', int64),
    ('height', int64),
    ('outputWidth', int64),
    ('outputHeight', int64),
    ('inputImage', uint8[:, :, :]),
    ('outputImage', uint8[:, :, :]),
    ('scale', float64),
]

@jitclass(spec)    
class EPI:
    def weight_cal(self, vs):
        gamma = 2
        x_gamma = gamma/2
        
        tmp = (1-vs)
        vc3 = (tmp*vs*vs)*x_gamma
        vc0 = (tmp*tmp*vs)*x_gamma
        vc1 = tmp + 2*vc0 - 1*vc3
        vc2 = vs + 2*vc3 - 1*vc0
        vc0 = (-1)*vc0
        vc3 = (-1)*vc3

        return vc0, vc1, vc2, vc3

    ##################
    #主要運算function
    ##################
    def bilinear(self, color):
        oriImage = self.inputImage[..., color]
        scale_factor_h = 0.5 #scale_factor_h的倒數 
        scale_factor_v = 0.5#scale_factor_v的倒數 
        
        for oy in range(1, self.outputHeight - 3):
            for ox in range(1, self.outputWidth - 3):
                x = (ox + 0.5) * (scale_factor_h) - 0.5
                y = (oy + 0.5) * (scale_factor_v) - 0.5
                
                ix = int(x)	#i = floor(x) 
                iy = int(y) #j = floor(y)
                dx = x - float(ix)
                dy = y - float(iy)
                
                a1, a2, a3, a4 = self.weight_cal(dy)

                in1 = oriImage[iy-1, ix-1] * a1 + oriImage[iy, ix-1] * a2 + \
                    oriImage[iy+1, ix-1] * a3 + oriImage[iy+2, ix-1] * a4
                    
                in2 = oriImage[iy-1,ix] * a1 + oriImage[iy,ix] * a2 + \
                    oriImage[iy+1, ix] * a3 + oriImage[iy+2, ix] * a4
                
                in3 = oriImage[iy-1, ix+1] * a1 + oriImage[iy, ix+1] * a2 + \
                    oriImage[iy+1, ix+1] * a3 + oriImage[iy+2, ix+1] * a4
                
                in4 = oriImage[iy-1, ix+2] * a1 + oriImage[iy, ix+2] * a2 + \
                    oriImage[iy+1, ix+2] * a3 + oriImage[iy+2, ix+2] * a4
                
                a1, a2, a3, a4 = self.weight_cal(dx)
                
                temp = round(in1*a1 + in2*a2 + in3*a3 + in4*a4)
                
                temp *= (temp>0)
                if temp > 255 :
                    temp = 255

                self.outputImage[oy, ox, color] = temp


    def __init__(self, oriImage, scale):
        self.width = oriImage.shape[1]
        self.height = oriImage.shape[0]

        self.scale = scale
        self.outputWidth = int(self.width * scale)
        self.outputHeight = int(self.height * scale)
        self.inputImage = np.copy(oriImage)
        self.outputImage = np.zeros((self.outputHeight, self.outputWidth, 3), dtype=np.uint8)

    def Algorithm(self):
        self.bilinear(0)
        self.bilinear(1)
        self.bilinear(2)

        return self.outputImage
        