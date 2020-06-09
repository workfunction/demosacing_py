import numpy as np
import numba as nb
from numba.experimental import jitclass
from numba import int64
from numba import uint8
from numba import float64

spec = [
    ('width', int64),
    ('height', int64),
    ('outputWidth', int64),
    ('outputHeight', int64),
    ('inputImage', uint8[:, :]),
    ('outputImage', uint8[:, :, :]),
    ('scale', float64),
    ('mode', uint8),
]

@jitclass(spec)    
class MEPL:
    def __init__(self, oriImage, scale):
        self.width = oriImage.shape[1]
        self.height = oriImage.shape[0]

        self.scale = scale
        self.outputWidth = int(self.width * scale)
        self.outputHeight = int(self.height * scale)
        self.inputImage = np.copy(oriImage)
        self.outputImage = np.zeros((self.outputHeight, self.outputWidth, 3), dtype=np.uint8)
        self.mode = 0
        
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

        return np.array([vc0, vc1, vc2, vc3])
    
    def weight_cal3(self, d):
        r = 4 + d * 4 - d * d
        vc1 = ((2 + d) * (2 - d))/r
        vc0 = (d * (2 + d))/r
        vc2 = (d * (2 - d))/r

        return np.array([vc0, vc1, vc2])
    
    def is_green(self, i, j):
        return (i%2) != (j%2)

    def _DeltaH(self, i, j):
        if self.is_green(i, j) == True:
            return (self.inputImage[i, j] / 2) + (self.inputImage[i, j-2] / 4) +   \
                (self.inputImage[i, j+2] / 4) - (self.inputImage[i, j-1] / 2) - \
                (self.inputImage[i, j+1] / 2)
        else:
            return (self.inputImage[i, j-1] / 2) + (self.inputImage[i, j+1] / 2) - \
                (self.inputImage[i, j-2] / 4) - (self.inputImage[i, j+2] / 4) - \
                (self.inputImage[i, j] / 2)

    def _DeltaV(self, i, j):
        if self.is_green(i, j) == True:
            return (self.inputImage[i, j] / 2) + (self.inputImage[i-2, j] / 4) +   \
                (self.inputImage[i+2, j] / 4) - (self.inputImage[i-1, j] / 2) - \
                (self.inputImage[i+1, j] / 2)
        else:
            return (self.inputImage[i-1, j] / 2) + (self.inputImage[i+1, j] / 2) - \
                (self.inputImage[i-2, j] / 4) - (self.inputImage[i+2, j] / 4) - \
                (self.inputImage[i, j] / 2)

    def DeltaH(self, i, j, dx, dy, ioffset=0, joffset=0):
        delta = np.empty((4, 4))

        for m in range(4):
            for n in range(4):
                delta[m, n] = self._DeltaH(i+(m-ioffset-1)*2, j+(n-joffset-1))

        a = self.weight_cal(dx)
        b = self.weight_cal(dy/2)

        delta_total = np.dot(delta, a)
        Delta_H_total = np.dot(delta_total, b)
        Dh_G = np.dot(np.abs(delta[:, 0] - delta[:, 3]), np.array([1/8,3/8,3/8,1/8]))

        return Delta_H_total, Dh_G

    def DeltaV(self, i, j, dy, dx, ioffset=0, joffset=0):
        delta = np.empty((4, 4))

        for m in range(4):
            for n in range(4):
                delta[m,n] = self._DeltaV(i+(n-ioffset-1), j+(m-joffset-1)*2)
        
        a = self.weight_cal(dy)
        b = self.weight_cal(dx/2)
        
        delta_total = np.dot(delta, a)
        Delta_V_total = np.dot(delta_total, b)
        Dv_G = np.dot(np.abs(delta[:, 0] - delta[:, 3]), np.array([1/8,3/8,3/8,1/8]))

        return Delta_V_total, Dv_G
    
    def getDelta(self, dh, dv, gh, gv):
        if gv <= gh:
            if gv * 4 <= gh:
                return dv
            elif gv * 2 <= gh:
                return (3 * dv + dh) / 4
            else:
                return (dv + dh) / 2
        else:
            if gh * 4 <= gv:
                return dh
            elif gh * 2 <= gv:
                return (3 * dh + dv) / 4
            else:
                return (dh + dv) / 2

    def Algorithm(self):
        scale_factor_h = 0.5 #scale_factor_h的倒數 
        scale_factor_v = 0.5#scale_factor_v的倒數 
        
        for oy in range(10, self.outputHeight - 10):
            for ox in range(10, self.outputWidth - 10):
                x = (ox + 0.5) * (scale_factor_h) - 0.5
                y = (oy + 0.5) * (scale_factor_v) - 0.5
                
                j = int(x)	#i = floor(x) 
                i = int(y) #j = floor(y)
                dx = x - float(j)
                dy = y - float(i)

                '''
                * mode 0: |B| G   * mode 1: |G| B
                           G  R              R  G
                
                * mode 2: |G| R   * mode 3: |R| G
                           B  G              G  B
                '''
                self.mode = (i % 2) << 1 | (j % 2)

                if self.mode == 0 or self.mode == 3:
                    Delta_H_C0, Dh_C0 = self.DeltaH(i, j, dx, dy)
                    Delta_V_C0, Dv_C0 = self.DeltaV(i, j, dy, dx)                

                    Delta_H_C1, Dh_C1 = self.DeltaH(i+1, j, dx, 1-dy, joffset=1)
                    Delta_V_C1, Dv_C1 = self.DeltaV(i, j+1, dy, 1-dx, ioffset=1)                
                
                else:
                    Delta_H_C0, Dh_C0 = self.DeltaH(i, j, dx, dy)
                    Delta_V_C1, Dv_C1 = self.DeltaV(i, j, dy, dx)                

                    Delta_H_C1, Dh_C1 = self.DeltaH(i+1, j, dx, 1-dy, joffset=1)
                    Delta_V_C0, Dv_C0 = self.DeltaV(i, j+1, dy, 1-dx, ioffset=1)
                
                Delta_C0 = self.getDelta(Delta_H_C0, Delta_V_C0, Dh_C0, Dv_C0)
                Delta_C1 = self.getDelta(Delta_H_C1, Delta_V_C1, Dh_C1, Dv_C1)

                a = self.weight_cal(dy)
                b = self.weight_cal(dx)
                
                arr = self.inputImage[i-1:i+3, j-1:j+3]
                float_arr = arr.astype(nb.float64)
                
                if self.mode == 0 or self.mode == 3:
                    det = np.array([[Delta_C1, 0 ,Delta_C1, 0],
                                    [0, Delta_C0, 0, Delta_C0],
                                    [Delta_C1, 0 ,Delta_C1, 0],
                                    [0, Delta_C0, 0, Delta_C0]], dtype=nb.float64)  
                else:
                    det = np.array([[0 ,Delta_C1, 0, Delta_C1],
                                    [Delta_C0, 0, Delta_C0, 0],
                                    [0 ,Delta_C1, 0, Delta_C1],
                                    [Delta_C0, 0, Delta_C0, 0]], dtype=nb.float64)

                float_arr = float_arr + det
                temp = np.dot(a.reshape(1,4), float_arr)
                epi  = np.dot(temp, b)
                
                tempG = np.sum(epi)
                
                if self.mode == 0:
                    tempB = tempG - Delta_C0
                    tempR = tempG - Delta_C1
                elif self.mode == 1:
                    tempB = tempG - Delta_C0
                    tempR = tempG - Delta_C1
                elif self.mode == 2:
                    tempR = tempG - Delta_C0
                    tempB = tempG - Delta_C1
                elif self.mode == 3:
                    tempR = tempG - Delta_C0
                    tempB = tempG - Delta_C1
                else:
                    tempR = 0
                    tempG = 0
                    tempB = 0

                self.outputImage[oy, ox, 1] = 255 if tempG > 255 else 0 if tempG < 0 else int(tempG)
                self.outputImage[oy, ox, 0] = 255 if tempR > 255 else 0 if tempR < 0 else int(tempR)
                self.outputImage[oy, ox, 2] = 255 if tempB > 255 else 0 if tempB < 0 else int(tempB)
                                    

        return self.outputImage
        