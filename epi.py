import numpy as np
import numba as nb

def weight_cal(vs):
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

@nb.njit
def bi(outputImage, oriImage, outputHeight, outputWidth):
    scale_factor_h = 0.5 #scale_factor_h的倒數 
    scale_factor_v = 0.5#scale_factor_v的倒數 
    
    for oy in range(1, outputHeight - 3):
        for ox in range(1, outputWidth - 3):
            x = (ox + 0.5) * (scale_factor_h) - 0.5
            y = (oy + 0.5) * (scale_factor_v) - 0.5
            
            ix = int(x)	#i = floor(x) 
            iy = int(y) #j = floor(y)
            dx = x - float(ix)
            dy = y - float(iy)
            
            #a1, a2, a3, a4 = weight_cal(dy)
            gamma = 2
            x_gamma = gamma/2
            
            tmp = (1-dy)
            a4 = (tmp*dy*dy)*x_gamma
            a1 = (tmp*tmp*dy)*x_gamma
            a2 = tmp + 2*a1 - 1*a4
            a3 = dy + 2*a4 - 1*a1
            a1 = (-1)*a1
            a4 = (-1)*a4

            in1 = oriImage[iy-1, ix-1] * a1 + oriImage[iy, ix-1] * a2 + \
                oriImage[iy+1, ix-1] * a3 + oriImage[iy+2, ix-1] * a4
                
            in2 = oriImage[iy-1,ix] * a1 + oriImage[iy,ix] * a2 + \
                oriImage[iy+1, ix] * a3 + oriImage[iy+2, ix] * a4
            
            in3 = oriImage[iy-1, ix+1] * a1 + oriImage[iy, ix+1] * a2 + \
                oriImage[iy+1, ix+1] * a3 + oriImage[iy+2, ix+1] * a4
            
            in4 = oriImage[iy-1, ix+2] * a1 + oriImage[iy, ix+2] * a2 + \
                oriImage[iy+1, ix+2] * a3 + oriImage[iy+2, ix+2] * a4
            
            #a1, a2, a3, a4 = weight_cal(dx)
            tmp = (1-dx)
            a4 = (tmp*dx*dx)*x_gamma
            a1 = (tmp*tmp*dx)*x_gamma
            a2 = tmp + 2*a1 - 1*a4
            a3 = dx + 2*a4 - 1*a1
            a1 = (-1)*a1
            a4 = (-1)*a4
            
            temp = round(in1*a1 + in2*a2 + in3*a3 + in4*a4)
            
            temp *= (temp>0)
            if temp > 255 :
                temp = 255

            outputImage[oy, ox] = temp
    
class EPI:
    ##################
    #主要運算function
    ##################
    def bilinear(self, oriImage):        
        bi(self.outputImage, oriImage, self.outputHeight, self.outputWidth)


    def __init__(self, oriImage, width, height, scale):
        self.width = width
        self.height = height
        self.scale = scale
        self.outputWidth = int(width * scale)
        self.outputHeight = int(height * scale)
        self.tmpImage = np.copy(oriImage)
        self.outputImage = np.zeros((self.outputHeight, self.outputWidth), int)

    def Algorithm(self):
        out = np.zeros((self.outputHeight, self.outputWidth, 3), int)
        self.bilinear(self.tmpImage[...,0])
        out[...,0] = self.outputImage
        self.bilinear(self.tmpImage[...,1])
        out[...,1] = self.outputImage
        self.bilinear(self.tmpImage[...,2])
        out[...,2] = self.outputImage
        return out
        